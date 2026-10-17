#!/usr/bin/env python3
"""
Flagship gallery pipeline: BiRefNet mask -> showroom composite -> color normalize.
The car's pixels are copied, never regenerated.

Usage (on a GitHub Actions runner or any machine that can reach the CDN):
  python pipeline.py --stock P1063A --manifest manifest.json --plate plate.jpg --out runs/P1063A
Optional: --model birefnet-general-lite  (faster), --overrides overrides.json  ({"<id>": "exterior"|"detail"})
"""
import argparse, json, time
from pathlib import Path
import numpy as np, requests
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageDraw, ImageFont
from rembg import new_session, remove

CDN = "https://imagescf.dealercenter.net/2048/1536/{id}.jpg"
HDR = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130 Safari/537.36",
       "Referer": "https://www.flagshipcarsandclassics.com/"}
OUT_W, OUT_H = 2048, 1365      # exact 3:2, same as the hero shots
FLOOR_Y   = 0.82               # tire contact line, fraction of height (matches heroes)
CAR_W     = 0.70               # car width as fraction of frame width
CAR_H_MAX = 0.62               # cap for tall/portrait shots
ERODE_PX  = 1                  # shave this many px off the mask edge to kill halos
MAX_UPSCALE = 1.3              # sharpness guard: small cars in portrait shots stay a bit smaller instead of going soft
SHADOW_OP = 0.62
REFL_OP   = 0.20


def fetch(pid, dest):
    p = dest / f"{pid}.jpg"
    if not p.exists():
        r = requests.get(CDN.format(id=pid), headers=HDR, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


# ---- whole-car vs part/interior: CLIP zero-shot (the mask alone cannot tell a wheel close-up from a car) ----
EXT_PROMPTS = ["a photo of a whole car seen from outside with the entire vehicle visible",
               "a photo of a parked sports car, full body in view, front, side or rear view"]
DET_PROMPTS = ["a close-up photo of a car wheel, tire or brake",
               "a close-up photo of a car headlight, taillight, badge, vent, mirror or body panel",
               "a photo of a car interior: dashboard, seats, steering wheel, door panel or center console",
               "a close-up photo of a gear shifter, start button, pedals or infotainment screen",
               "a photo of a car engine bay, open trunk, floor mat or car cover",
               "a photo of a window sticker, document or paperwork"]
CLIP_MIN = 0.50


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval(); proc = CLIPProcessor.from_pretrained(name)
    return model, proc


def clip_exterior_prob(img, ctx):
    """P(whole car) = softmax mass on the whole-car prompts. Uses logits_per_image (stable across transformers versions)."""
    import torch
    model, proc = ctx
    with torch.no_grad():
        inp = proc(text=EXT_PROMPTS + DET_PROMPTS, images=img, return_tensors="pt", padding=True)
        p = model(**inp).logits_per_image.softmax(-1)[0]
    return float(p[:len(EXT_PROMPTS)].sum())


def mask_stats(mask, clip_p):
    """Returns (kind, flags, frac). 'exterior' only if CLIP says whole car AND the mask is a plausible, uncropped car."""
    a = np.asarray(mask) > 127
    h, w = a.shape
    flags = [f"clip={clip_p:.2f}"]
    frac = float(a.mean())
    if clip_p < CLIP_MIN: return "detail", flags + ["not-whole-car"], frac
    if frac < 0.05: return "detail", flags + ["tiny"], frac
    ys, xs = np.where(a)
    bw = (xs.max() - xs.min()) / w; bh = (ys.max() - ys.min()) / h
    if frac > 0.80 or (bw > 0.97 and bh > 0.97):
        return "detail", flags + ["whole-frame"], frac
    if bw < 0.35:
        return "detail", flags + ["tiny"], frac
    # edge contact measured against the CAR's own size: a whole car barely touches the frame (a mirror tip at most);
    # a cropped car has a straight cut along an edge -> keep it as a real photo (color only)
    cw, ch = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    top, bot = a[:3].any(0).sum() / cw, a[-3:].any(0).sum() / cw
    left, right = a[:, :3].any(1).sum() / ch, a[:, -3:].any(1).sum() / ch
    if bot > 0.04 or top > 0.06 or left > 0.08 or right > 0.08:
        return "detail", flags + [f"cropped(b{bot:.2f} l{left:.2f} r{right:.2f} t{top:.2f})"], frac
    if xs.min() <= 2 or xs.max() >= w - 3: flags.append("edge-touch")
    # holes: filled-bbox area vs mask area inside bbox
    box = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if 1 - box.mean() > 0.45: flags.append("holes")
    return "exterior", flags, frac


def neutralize(img, mask=None, strength=0.5, cap=0.08):
    """Mild gray-world white balance, computed on the car pixels only when a mask is given."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    sel = arr if mask is None else arr[np.asarray(mask) > 127]
    means = sel.reshape(-1, 3).mean(0); g = means.mean()
    gains = np.clip(g / means, 1 - cap, 1 + cap)
    gains = 1 + (gains - 1) * strength
    return Image.fromarray(np.clip(arr * gains, 0, 255).astype(np.uint8))


def ground_line(alpha):
    """Tire contact points: the lowest car pixel in the left third and in the right third of the car.
    Returns (xl, yl, xr, yr) in cut coordinates; falls back to a level line at the lowest point."""
    a = alpha > 128; h, w = a.shape
    has = a.any(0); bottom = np.where(has, h - 1 - np.argmax(a[::-1], axis=0), -1)
    third = max(1, w // 3)
    xl = int(np.argmax(bottom[:third])); xr = w - third + int(np.argmax(bottom[w - third:]))
    yl, yr = int(bottom[xl]), int(bottom[xr])
    if yl < 0 or yr < 0 or abs(yl - yr) > 0.25 * h or xr - xl < 0.3 * w:
        yb = int(bottom.max()); return 0, yb, w - 1, yb
    return xl, yl, xr, yr


def composite(img, mask, plate):
    m = mask
    for _ in range(ERODE_PX): m = m.filter(ImageFilter.MinFilter(3))
    m = m.filter(ImageFilter.GaussianBlur(0.8))
    cut = neutralize(img, mask).convert("RGBA"); cut.putalpha(m)
    bbox = m.getbbox(); cut = cut.crop(bbox)
    s = min(CAR_W * OUT_W / cut.width, CAR_H_MAX * OUT_H / cut.height, MAX_UPSCALE)   # never blow up a small car past 1.3x
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    cw, ch = cut.size
    A = np.asarray(cut.getchannel("A")).astype(np.float32)
    xl, yl, xr, yr = ground_line(A)
    floor = yl + (yr - yl) * (np.arange(cw) - xl) / max(1, xr - xl)          # ground line under every column
    x = (OUT_W - cw) // 2
    y = round(FLOOR_Y * OUT_H - (yl + yr) / 2)                                # tire line, not the bbox, sits on FLOOR_Y
    out = plate.convert("RGBA").copy()

    # 1) floor reflection: each column mirrored about the ground line (correct for 3/4 views), fading out quickly
    rh = int(ch * 0.45); H2 = ch + rh
    ys = np.arange(H2)[:, None].astype(np.float32)
    src_y = np.round(2 * floor[None, :] - ys).astype(int)
    below = ys > floor[None, :]
    valid = below & (src_y >= 0) & (src_y < ch)
    C = np.asarray(cut).astype(np.float32)
    sy = np.clip(src_y, 0, ch - 1)
    R = np.take_along_axis(C, np.repeat(sy[:, :, None], 4, axis=2), axis=0)
    fade = np.clip(1 - (ys - floor[None, :]) / (ch * 0.30), 0, 1) ** 1.5
    R[..., 3] = np.where(valid, R[..., 3] * fade * REFL_OP, 0)
    refl = Image.fromarray(np.clip(R, 0, 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(max(2, cw * 0.002)))
    out.alpha_composite(refl, (x, y))

    # 2) shadows: a soft ambient band along the ground line + tight contact pads under the two tires
    sh = Image.new("L", (cw + 400, H2 + 200), 0); d = ImageDraw.Draw(sh); ox, oy = 200, 100
    band = ch * 0.035; ext = cw * 0.04
    d.polygon([(ox + xl - ext, oy + yl - band * 0.6), (ox + xr + ext, oy + yr - band * 0.6),
               (ox + xr + ext, oy + yr + band * 0.5), (ox + xl - ext, oy + yl + band * 0.5)], fill=int(255 * SHADOW_OP))
    sh = sh.filter(ImageFilter.GaussianBlur(max(4, cw * 0.018)))
    pad = Image.new("L", sh.size, 0); dp = ImageDraw.Draw(pad); pw, ph = cw * 0.07, ch * 0.018
    for (px, py) in ((xl, yl), (xr, yr)):
        dp.ellipse([ox + px - pw, oy + py - ph, ox + px + pw, oy + py + ph * 0.8], fill=235)
    pad = pad.filter(ImageFilter.GaussianBlur(max(2, cw * 0.006)))
    sha = ImageChops.lighter(sh, pad)
    shadow = Image.new("RGBA", sha.size, (12, 12, 14, 255)); shadow.putalpha(sha)
    out.alpha_composite(shadow, (x - ox, y - oy))

    # 3) the car itself, pixels untouched apart from the mild white balance
    out.alpha_composite(cut, (x, y))
    return out.convert("RGB")


def normalize_detail(img):
    im = neutralize(img, None, strength=0.4, cap=0.06)
    return ImageOps.autocontrast(im, cutoff=0.5)


def contact_sheet(rows, qa_dir, per_sheet=8):
    """Before|after pairs, 2 columns, 8 pairs per sheet -> qa/sheet-01.jpg, sheet-02.jpg, ... (readable at screen size)."""
    W, H, pad, cols = 420, 280, 6, 2
    try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception: f = ImageFont.load_default()
    names = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        sheet = Image.new("RGB", (cols * (2 * W + 3 * pad), ((len(chunk) + 1) // cols) * (H + 30 + pad)), "white")
        d = ImageDraw.Draw(sheet)
        for i, (label, before, after) in enumerate(chunk):
            cx = (i % cols) * (2 * W + 3 * pad); cy = (i // cols) * (H + 30 + pad)
            d.text((cx + pad, cy + 6), label, fill=(20, 20, 20), font=f)
            sheet.paste(ImageOps.contain(before, (W, H)), (cx + pad, cy + 28))
            sheet.paste(ImageOps.contain(after, (W, H)), (cx + W + 2 * pad, cy + 28))
        name = f"sheet-{s // per_sheet + 1:02d}.jpg"; sheet.save(qa_dir / name, quality=80); names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True); ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--plate", default="plate.jpg"); ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="birefnet-general"); ap.add_argument("--overrides", default="overrides.json")
    a = ap.parse_args()
    man = json.load(open(a.manifest))[a.stock]
    ov = json.load(open(a.overrides)) if Path(a.overrides).exists() else {}
    root = Path(a.out or f"runs/{a.stock}"); orig = root / "originals"; out = root / "out"; qa = root / "qa"
    for p in (orig, out, qa): p.mkdir(parents=True, exist_ok=True)
    plate = Image.open(a.plate).convert("RGB")
    assert plate.size == (OUT_W, OUT_H), f"plate must be {OUT_W}x{OUT_H}, got {plate.size}"
    session = new_session(a.model)
    clipctx = load_clip()
    report = {"stock": a.stock, "model": a.model, "images": []}
    rows = []
    ids = [i for i in man["ids"] if i != man["lead"]]      # the hero stays #1 and is never reprocessed
    for n, pid in enumerate(ids, start=1):
        t0 = time.time()
        src = Image.open(fetch(pid, orig)).convert("RGB")
        clip_p = clip_exterior_prob(src, clipctx)
        if pid in ov and ov[pid] == "detail":
            mask = None; kind, flags, frac = "detail", [f"clip={clip_p:.2f}", "override"], 0.0
        elif clip_p < CLIP_MIN and ov.get(pid) != "exterior":       # close-up / interior: skip the mask entirely
            mask = None; kind, flags, frac = "detail", [f"clip={clip_p:.2f}", "not-whole-car"], 0.0
        else:
            mask = remove(src, session=session, only_mask=True, post_process_mask=True)
            kind, flags, frac = mask_stats(mask, clip_p)
            if pid in ov: kind = ov[pid]; flags.append("override")
        if kind == "exterior": result = composite(src, mask, plate)
        else: result = normalize_detail(src)
        name = f"{n:03d}-{kind}-{pid[-8:]}.jpg"
        result.save(out / name, quality=93, subsampling=0)
        report["images"].append({"n": n, "id": pid, "kind": kind, "flags": flags, "mask_frac": round(frac, 3),
                                 "file": name, "sec": round(time.time() - t0, 1)})
        rows.append((f"{n:03d} {kind} {' '.join(flags)}", src, result))
        print(f"{n:03d} {pid} {kind} {flags} {time.time()-t0:.1f}s", flush=True)
    report["sheets"] = contact_sheet(rows, qa)
    report["lead"] = man["lead"]; report["count"] = len(ids)
    json.dump(report, open(root / "report.json", "w"), indent=1)
    print("done", len(ids), "images")


if __name__ == "__main__":
    main()
