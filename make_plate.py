#!/usr/bin/env python3
"""
Build the clean 2048x1365 showroom plate from backdrop-3.jpg (the 2576x1932 landscape showroom photo).
Deterministic: no generative step. Order: crop -> scale -> clean -> extend floor.
Removed so nothing but the listed car shares the frame: the other car + glass wall + side prints (flat wall
sampled from the real wall), the yellow tape marks (found by color), the photographer's mirror. The floor is
extended downward by stretching the nearest floor rows 1.35x (seamless on marble) so the wall/floor seam sits at 59% like the heroes.
Usage: python make_plate.py backdrop-3.jpg plate.jpg
"""
import sys
import numpy as np
from PIL import Image

SRC, DST = sys.argv[1], sys.argv[2]
OUT_W, OUT_H = 2048, 1365
PRINT_TOP, SEAM_SRC, PRINT_CX = 425, 1404, 1262        # measured in backdrop-3 (2576x1932)
TARGET_SEAM, TARGET_TOP = 0.59, 0.02                    # same vertical layout as the hero shots

src = Image.open(SRC).convert("RGB"); W, H = src.size
win_h = round((SEAM_SRC - PRINT_TOP) / (TARGET_SEAM - TARGET_TOP)); win_w = round(win_h * 3 / 2)
top = PRINT_TOP - round(TARGET_TOP * win_h)
left = 0 if win_w >= W else max(0, min(W - win_w, PRINT_CX - win_w // 2))
crop = src.crop((left, top, min(W, left + win_w), H))
s = OUT_W / crop.width
work = crop.resize((OUT_W, round(crop.height * s)), Image.LANCZOS)      # plate scale, floor still short
a = np.asarray(work).astype(np.float32); g = a.mean(2); h0 = a.shape[0]

# pillars + seam (detected; fallbacks are the measured values)
frac = (g[150:700] < 70).mean(0); dark = np.where(frac > 0.85)[0]
runs, st, pv = [], dark[0], dark[0]
for x in dark[1:]:
    if x != pv + 1: runs.append((st, pv)); st = x
    pv = x
runs.append((st, pv)); runs = [r for r in runs if r[1] - r[0] > 30]
L0, L1 = next((r for r in runs if r[1] < OUT_W // 2), (346, 434))
R0, R1 = next((r for r in reversed(runs) if r[0] > OUT_W // 2), (1575, 1640))
band = g[int(h0 * .5): int(h0 * .85), L1 + 40: R0 - 40].mean(1)
seam = int(h0 * .5) + int(np.argmax(np.abs(np.diff(band))))
print(f"pillars L {L0}-{L1}  R {R0}-{R1}  seam {seam}  work height {h0}")

out = a.copy(); rng = np.random.default_rng(0)
# 1. flat wall outside the pillars down to just below the baseboard, sampled per row beside the print
for y in range(0, seam + 1):   # wall rows (incl. baseboard): sample the real wall just outside the print
    out[y, :L0] = a[y, L1 + 40: L1 + 140].mean(0); out[y, R1 + 1:] = a[y, R0 - 140: R0 - 40].mean(0)
out[:seam + 1, :L0] += rng.normal(0, 1.5, (seam + 1, L0, 1)); out[:seam + 1, R1 + 1:] += rng.normal(0, 1.5, (seam + 1, OUT_W - R1 - 1, 1))

def patch(x0, y0, x1, y1, sx, feather_top=True):
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(OUT_W, x1), min(h0, y1)
    h, w = y1 - y0, x1 - x0; sx = max(0, min(sx, OUT_W - w))
    srcb = out[y0:y1, sx:sx + w].copy(); m = np.ones((h, w, 1), np.float32); f = 14
    for i in range(f):
        if feather_top: m[i, :] *= (i + 1) / f
        m[h - 1 - i, :] *= (i + 1) / f; m[:, i] *= (i + 1) / f; m[:, w - 1 - i] *= (i + 1) / f
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - m) + srcb * m

# 2. floor reflection of the removed car (left), 3. photographer's mirror (bottom right)
patch(0, seam + 1, L0 + 40, seam + 230, 1000, feather_top=False)
patch(1500, h0 - 230, OUT_W, h0, 760)
# 4. yellow tape marks, found by color
yel = (a[..., 0] > 140) & (a[..., 1] > 110) & (a[..., 2] < 110) & (a[..., 0] - a[..., 2] > 70); yel[:seam] = False
boxes = []
for y, x in zip(*np.where(yel)):
    for b in boxes:
        if abs(b[0] - x) < 120 and abs(b[1] - y) < 60:
            b[2], b[3], b[4], b[5] = max(b[2], x), max(b[3], y), min(b[4], x), min(b[5], y); break
    else: boxes.append([x, y, x, y, x, y])
for cx, cy, mx, my, nx, ny in boxes:
    patch(int(nx) - 30, int(ny) - 25, int(mx) + 30, int(my) + 25, int(nx) + 160 if nx < OUT_W - 400 else int(nx) - 400)
print("tape marks patched:", len(boxes))

# 5. extend the floor to OUT_H: stretch the bottom floor rows vertically (seamless on marble; no copy join)
need = OUT_H - h0
final = np.zeros((OUT_H, OUT_W, 3), np.float32); final[:h0] = out
if need > 0:
    STRETCH_ROWS = 420                                       # floor rows nearest the camera
    band = Image.fromarray(np.clip(out[h0 - STRETCH_ROWS: h0], 0, 255).astype(np.uint8))
    band = band.resize((OUT_W, STRETCH_ROWS + need), Image.LANCZOS)
    final[h0 - STRETCH_ROWS:] = np.asarray(band).astype(np.float32)
Image.fromarray(np.clip(final, 0, 255).astype(np.uint8)).save(DST, quality=95, subsampling=0)
print("wrote", DST, OUT_W, "x", OUT_H, "floor extended by", need)
