import numpy as np
from PIL import Image

im = Image.open('/mnt/user-data/uploads/result.png').convert('L')
arr = np.array(im)

row_bounds = [73, 92, 111, 131, 150, 170, 189, 209, 228, 247, 267, 286, 306, 325, 345,
              364, 383, 403, 422, 442, 461, 481, 500, 519, 539, 558, 578, 597, 617, 636,
              655, 675, 694, 714, 733, 753, 772, 791, 811, 830, 850, 869, 889, 908, 927,
              947, 966, 986, 1005, 1025, 1044]
sl_x0, sl_x1 = 14, 43

sl_values = list(range(50, 0, -1))  # row 0 (top) = SL 50, row 49 (bottom) = SL 1

TARGET = 22  # standardized glyph size

def tight_crop(bin_img):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return None
    return bin_img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def resize_glyph(glyph_bin):
    g = Image.fromarray((glyph_bin*255).astype(np.uint8))
    g = g.resize((TARGET, TARGET), Image.LANCZOS)
    return np.array(g) / 255.0

X, y = [], []
skipped = []
for i, sl_val in enumerate(sl_values):
    y0, y1 = row_bounds[i]+1, row_bounds[i+1]-1
    crop = arr[y0:y1, sl_x0:sl_x1]
    bw = (crop < 150).astype(np.uint8)
    tc = tight_crop(bw)
    if tc is None:
        skipped.append(sl_val); continue
    digits = str(sl_val)
    n = len(digits)
    w = tc.shape[1]
    # equal-width slicing across the tight ink bounding box
    edges = np.linspace(0, w, n+1).astype(int)
    ok = True
    glyphs = []
    for d in range(n):
        piece = tc[:, edges[d]:edges[d+1]]
        pt = tight_crop(piece)
        if pt is None or pt.shape[0] < 3 or pt.shape[1] < 2:
            ok = False
            break
        glyphs.append(pt)
    if not ok:
        skipped.append(sl_val); continue
    for ch, g in zip(digits, glyphs):
        X.append(resize_glyph(g))
        y.append(ch)

print(f"Harvested {len(X)} labeled digit glyphs from {len(sl_values)} SL cells")
print(f"Skipped (segmentation failed): {skipped}")
from collections import Counter
print("Class distribution:", Counter(y))

np.save('digit_X.npy', np.array(X))
np.save('digit_y.npy', np.array(y))
