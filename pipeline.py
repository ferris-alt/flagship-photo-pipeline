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
SHADOW_OP = 0.55
REFL_OP   = 0.28


def fetch(pid, dest):
    p = dest / f"{pid}.jpg"
    if not p.exists():
        r = requests.get(CDN.format(id=pid), headers=HDR, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def mask_stats(mask):
    a = np.asarray(mask) > 127
    h, w = a.shape
    flags = []
    frac = float(a.mean())
    if frac < 0.05: return "detail", flags + ["tiny"], frac
    ys, xs = np.where(a)
    bw = (xs.max() - xs.min()) / w; bh = (ys.max() - ys.min()) / h
    if frac > 0.80 or (bw > 0.97 and bh > 0.97):
        return "detail", flags + ["whole-frame"], frac
    if bw < 0.35:
        return "detail", flags + ["tiny"], frac
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


def composite(img, mask, plate):
    m = mask
    for _ in range(ERODE_PX): m = m.filter(ImageFilter.MinFilter(3))
    m = m.filter(ImageFilter.GaussianBlur(0.8))
    cut = neutralize(img, mask).convert("RGBA"); cut.putalpha(m)
    bbox = m.getbbox(); cut = cut.crop(bbox)
    s = min(CAR_W * OUT_W / cut.width, CAR_H_MAX * OUT_H / cut.height)
    cut = cut.resize((round(cut.width * s), round(cut.height * s)), Image.LANCZOS)
    x = (OUT_W - cut.width) // 2
    y = round(FLOOR_Y * OUT_H) - cut.height
    out = plate.convert("RGBA").copy()
    # floor reflection: flipped, fading, slightly blurred
    refl = ImageOps.flip(cut)
    grad = ImageOps.invert(Image.linear_gradient("L").resize(refl.size)).point(lambda v: int(v * REFL_OP))
    refl.putalpha(ImageChops.multiply(refl.getchannel("A"), grad))
    refl = refl.filter(ImageFilter.GaussianBlur(2))
    out.alpha_composite(refl, (x, y + cut.height))
    # contact shadow: squashed alpha, blurred, under the tires
    sh = cut.getchannel("A").resize((cut.width, max(8, int(cut.height * 0.12))))
    sh = sh.filter(ImageFilter.GaussianBlur(max(2, cut.width * 0.015)))
    shadow = Image.new("RGBA", sh.size, (0, 0, 0, 255)); shadow.putalpha(sh.point(lambda v: int(v * SHADOW_OP)))
    out.alpha_composite(shadow, (x, y + cut.height - sh.height // 2))
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
    report = {"stock": a.stock, "model": a.model, "images": []}
    rows = []
    ids = [i for i in man["ids"] if i != man["lead"]]      # the hero stays #1 and is never reprocessed
    for n, pid in enumerate(ids, start=1):
        t0 = time.time()
        src = Image.open(fetch(pid, orig)).convert("RGB")
        mask = remove(src, session=session, only_mask=True, post_process_mask=True)
        kind, flags, frac = mask_stats(mask)
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
