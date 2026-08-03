"""
Synthetic + real character dataset generator for the custom stencil-font
recognizer.

Since we don't have (and won't get) the exact source font, this renders
characters with the boldest available system monospace font and then
algorithmically carves stencil-style gaps into the strokes -- the same
"breaks in enclosed loops" pattern we measured directly on real harvested
glyphs from the document (see cell_r25_c2.png / cell_r26_c2.png analysis
and the SL-column digit harvest). This is a calibrated approximation, not
a guess: the gap width/density below is tuned to roughly match what we
measured, and is trivially swappable for a real font later (see FONT_PATH).

Output: a directory of labeled (image, label) pairs combining:
  - bulk synthetic renders (many per class, cheap, approximate font)
  - the real harvested glyphs (few per class, ground-truth accurate),
    oversampled so they carry meaningful weight during training despite
    being a small fraction of total volume

Usage:
    python3 synth_data.py --out data/ --per-class 300
"""
import argparse
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Config -- adjust FONT_PATH if/when a real matching font becomes available.
# Everything downstream (training script) is agnostic to this choice.
# ---------------------------------------------------------------------------
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
FONT_RENDER_SIZE = 64          # render large, then downscale -- crisper edges
GLYPH_SIZE = 32                 # final training image size (square)
CHARSET = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.:-,()/")

STENCIL_GAP_COUNT_RANGE = (1, 3)     # how many gap cuts per glyph render
STENCIL_GAP_WIDTH_RANGE = (1, 2)     # gap width in px at FONT_RENDER_SIZE scale
# NOTE: calibrate these against real glyphs if you get a chance to measure
# more precisely -- current values are a rough match to what was visually
# observed in the "PIPE ASSY..." crops (gaps roughly 1-2px wide at native
# scan resolution, 1-3 per complex glyph like A/B/P/R/O).


def render_base_glyph(ch, font):
    """Render one character to a tight-cropped binary numpy array (1=ink)."""
    canvas = Image.new("L", (FONT_RENDER_SIZE * 2, FONT_RENDER_SIZE * 2), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((FONT_RENDER_SIZE // 2, FONT_RENDER_SIZE // 2), ch,
               fill=255, font=font)
    arr = np.array(canvas)
    ys, xs = np.where(arr > 100)
    if len(xs) == 0:
        return None
    pad = 4
    y0, y1 = max(0, ys.min() - pad), ys.max() + pad
    x0, x1 = max(0, xs.min() - pad), xs.max() + pad
    return (arr[y0:y1, x0:x1] > 100).astype(np.uint8)


def punch_stencil_gaps(glyph_bin, rng):
    """Carve 1-3 short gaps through the glyph's strokes to mimic stencil-cut
    lettering. Purely geometric: pick a random point ON existing ink, cut a
    short straight segment through it at a random angle. Only "enclosed"
    or thick-stroke glyphs actually need this in reality, but applying it
    generically (including to already-open glyphs like I/L/1) just
    occasionally trims an edge, which is harmless variety, not wrong."""
    h, w = glyph_bin.shape
    out = glyph_bin.copy()
    ys, xs = np.where(out > 0)
    if len(xs) == 0:
        return out
    n_gaps = rng.integers(*STENCIL_GAP_COUNT_RANGE, endpoint=True)
    for _ in range(n_gaps):
        idx = rng.integers(0, len(xs))
        cx, cy = xs[idx], ys[idx]
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(2, min(h, w) * 0.35)
        gap_w = rng.integers(*STENCIL_GAP_WIDTH_RANGE, endpoint=True)
        dx, dy = np.cos(angle), np.sin(angle)
        x0, y0 = int(cx - dx * length / 2), int(cy - dy * length / 2)
        x1, y1 = int(cx + dx * length / 2), int(cy + dy * length / 2)
        cv2.line(out, (x0, y0), (x1, y1), color=0, thickness=int(gap_w))
    return out


def to_fixed_size(glyph_bin, size=GLYPH_SIZE):
    img = Image.fromarray((glyph_bin * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img) / 255.0


def augment_one(img, rng):
    """Same augmentation family used earlier on the real harvested glyphs --
    reused here so synthetic and real data go through an identical pipeline
    and the model can't trivially tell them apart by artifact signature."""
    h, w = img.shape
    angle = rng.uniform(-6, 6)
    scale = rng.uniform(0.85, 1.15)
    tx, ty = rng.uniform(-2, 2), rng.uniform(-2, 2)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine((img * 255).astype(np.uint8), M, (w, h),
                          borderValue=0, flags=cv2.INTER_LINEAR)
    k = rng.choice([1, 2])
    kernel = np.ones((k, k), np.uint8)
    if rng.random() < 0.5:
        out = cv2.dilate(out, kernel, iterations=1)
    else:
        out = cv2.erode(out, kernel, iterations=1)
    sigma = rng.uniform(0.2, 1.1)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    noise = rng.normal(0, rng.uniform(3, 12), size=out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255)
    return out / 255.0


def load_real_glyphs():
    """Pull in the real harvested glyphs (digits from harvest_digits.py +
    letters from harvest_letters.py, combined). Returns (X, y) or
    (None, None) if not found."""
    if os.path.exists("real_X.npy") and os.path.exists("real_y.npy"):
        return np.load("real_X.npy"), np.load("real_y.npy")
    # fall back to digits-only if the combined file wasn't built yet
    if os.path.exists("digit_X.npy") and os.path.exists("digit_y.npy"):
        return np.load("digit_X.npy"), np.load("digit_y.npy")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-class", type=int, default=300,
                     help="synthetic images per character class")
    ap.add_argument("--real-oversample", type=int, default=20,
                     help="how many augmented copies to generate per real "
                          "harvested glyph, so real (accurate) data carries "
                          "meaningful weight against bulk synthetic data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    font = ImageFont.truetype(FONT_PATH, FONT_RENDER_SIZE)

    all_X, all_y, all_src = [], [], []  # src: 'synthetic' or 'real'

    print(f"Rendering base glyphs from {FONT_PATH} ...")
    base_glyphs = {}
    for ch in CHARSET:
        g = render_base_glyph(ch, font)
        if g is None:
            print(f"  WARNING: font has no glyph for '{ch}', skipping class")
            continue
        base_glyphs[ch] = g

    print(f"Generating {args.per_class} synthetic samples per class "
          f"({len(base_glyphs)} classes) ...")
    for ch, base in base_glyphs.items():
        for _ in range(args.per_class):
            gapped = punch_stencil_gaps(base, rng)
            fixed = to_fixed_size(gapped)
            aug = augment_one(fixed, rng)
            all_X.append(aug)
            all_y.append(ch)
            all_src.append("synthetic")

    real_X, real_y = load_real_glyphs()
    if real_X is not None:
        print(f"Blending in {len(real_X)} real harvested glyphs, "
              f"{args.real_oversample}x augmented copies each ...")
        for img, label in zip(real_X, real_y):
            resized = to_fixed_size(img)
            for _ in range(args.real_oversample):
                aug = augment_one(resized, rng)
                all_X.append(aug)
                all_y.append(str(label))
                all_src.append("real")
    else:
        print("No digit_X.npy/digit_y.npy found -- proceeding with "
              "synthetic-only data. Run harvest_digits.py first to include "
              "real anchor glyphs (recommended).")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)
    src = np.array(all_src)

    np.save(os.path.join(args.out, "X.npy"), X)
    np.save(os.path.join(args.out, "y.npy"), y)
    np.save(os.path.join(args.out, "src.npy"), src)

    print(f"Saved {len(X)} total samples to {args.out}/ "
          f"({(src == 'synthetic').sum()} synthetic, "
          f"{(src == 'real').sum()} real-derived)")
    print("Classes:", sorted(set(y)))


if __name__ == "__main__":
    main()
