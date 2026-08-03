"""
Harvest MORE real digit glyphs from two additional known-ground-truth
sources, to reinforce the digit classes (0,1,3,4,6,7,8,9) that scored
0% in the last held-out eval despite the class existing -- those failures
were data-scarcity, not confusability (unlike '1' vs 'I'/'L', which is a
separate, resolution-related issue this script does NOT fix).

Source 1: QTY column -- "1" for every row except SL 50 ("34").
Source 2: PIPE CUT LENGTH column, rows 42-48 (the only rows where it's
    populated) -- 7 distinct 4-digit values, verified in
    extracted_corrected.xlsx: 1227, 1268, 5258, 1141, 2630, 256, 3656.

Usage:
    python3 harvest_more_digits.py
"""
import numpy as np
from PIL import Image

IMAGE_PATH = "result.png"

ROW_BOUNDS = [73, 92, 111, 131, 150, 170, 189, 209, 228, 247, 267, 286, 306,
              325, 345, 364, 383, 403, 422, 442, 461, 481, 500, 519, 539,
              558, 578, 597, 617, 636, 655, 675, 694, 714, 733, 753, 772,
              791, 811, 830, 850, 869, 889, 908, 927, 947, 966, 986, 1005,
              1025, 1044]
QTY_X0, QTY_X1 = 325, 350        # inset from grid lines (321, 354)
LENGTH_X0, LENGTH_X1 = 357, 448  # inset from grid lines (354, 451)
INK_THRESHOLD = 150
TARGET = 22

sl_values = list(range(50, 0, -1))  # row i -> SL (50 - i)

QTY_GROUND_TRUTH = {sl: ("34" if sl == 50 else "1") for sl in range(1, 51)}
LENGTH_GROUND_TRUTH = {48: "1227", 47: "1268", 46: "5258", 45: "1141",
                        44: "2630", 43: "256", 42: "3656"}


def tight_crop(bin_img):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return None
    return bin_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def resize_glyph(glyph_bin):
    g = Image.fromarray((glyph_bin * 255).astype(np.uint8))
    g = g.resize((TARGET, TARGET), Image.LANCZOS)
    return np.array(g) / 255.0


def harvest_column(arr, x0, x1, ground_truth):
    X, y = [], []
    skipped = []
    for i, sl_val in enumerate(sl_values):
        if sl_val not in ground_truth:
            continue
        digits = ground_truth[sl_val]
        y0, y1 = ROW_BOUNDS[i] + 1, ROW_BOUNDS[i + 1] - 1
        crop = arr[y0:y1, x0:x1]
        bw = (crop < INK_THRESHOLD).astype(np.uint8)
        tc = tight_crop(bw)
        if tc is None:
            skipped.append(sl_val); continue
        n = len(digits)
        w = tc.shape[1]
        edges = np.linspace(0, w, n + 1).astype(int)
        glyphs = []
        ok = True
        for d in range(n):
            piece = tc[:, edges[d]:edges[d + 1]]
            pt = tight_crop(piece)
            if pt is None or pt.shape[0] < 3 or pt.shape[1] < 2:
                ok = False; break
            glyphs.append(pt)
        if not ok:
            skipped.append(sl_val); continue
        for ch, g in zip(digits, glyphs):
            X.append(resize_glyph(g))
            y.append(ch)
    return X, y, skipped


def main():
    im = Image.open(IMAGE_PATH).convert("L")
    arr = np.array(im)

    qty_X, qty_y, qty_skipped = harvest_column(arr, QTY_X0, QTY_X1, QTY_GROUND_TRUTH)
    print(f"QTY column: harvested {len(qty_X)} glyphs, skipped {qty_skipped}")

    len_X, len_y, len_skipped = harvest_column(arr, LENGTH_X0, LENGTH_X1, LENGTH_GROUND_TRUTH)
    print(f"Pipe Cut Length column: harvested {len(len_X)} glyphs, skipped {len_skipped}")

    X = np.array(qty_X + len_X)
    y = np.array(qty_y + len_y)
    from collections import Counter
    print(f"Total new digit glyphs: {len(X)}")
    print("Class distribution:", Counter(y))

    np.save("more_digit_X.npy", X)
    np.save("more_digit_y.npy", y)


if __name__ == "__main__":
    main()
