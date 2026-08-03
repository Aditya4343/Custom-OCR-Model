"""
Objectively score multiple candidate font files against the REAL harvested
glyphs (real_X.npy/real_y.npy), instead of eyeballing. For each font and
each character class we have real examples of, this renders the font's
clean glyph (no augmentation, no stencil-gap punching -- pure shape) and
compares it against the AVERAGE of that class's real glyphs using
normalized cross-correlation after center-of-mass alignment (to tolerate
small positional offsets between the render and the harvested crop).

This tells you which font's letterforms are closest to the real document's
font, before spending time on a full synthetic-data + training run.

Usage:
    mkdir font_candidates/
    # put candidate .ttf files in font_candidates/, e.g.:
    #   font_candidates/osifont-regular.ttf
    #   font_candidates/FreeMonoBold.ttf
    #   font_candidates/LiberationMono-Bold.ttf
    python3 font_compare.py --fonts-dir font_candidates/
"""
import argparse
import glob
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FONT_RENDER_SIZE = 64
GLYPH_SIZE = 32


def render_glyph(ch, font):
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
    tight = (arr[y0:y1, x0:x1] > 100).astype(np.float32)
    img = Image.fromarray((tight * 255).astype(np.uint8)).resize(
        (GLYPH_SIZE, GLYPH_SIZE), Image.LANCZOS)
    return np.array(img) / 255.0


def center_of_mass_align(img, size=GLYPH_SIZE):
    """Shift img so its ink center-of-mass sits at the image center --
    real crops and font renders won't be perfectly registered otherwise,
    which would tank correlation scores for reasons unrelated to actual
    shape similarity."""
    ys, xs = np.where(img > 0.1)
    if len(xs) == 0:
        return img
    cy, cx = ys.mean(), xs.mean()
    dy, dx = size / 2 - cy, size / 2 - cx
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img.astype(np.float32), M, (size, size))


def normalized_cross_correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-8:
        return 0.0
    return float((a * b).sum() / denom)


def is_monospace(font, tolerance=0.02):
    """Check if a font is monospace by comparing the rendered advance
    width of a narrow character ('i') against a wide one ('M'). We
    already confirmed by direct pixel measurement (character pitch
    calibration against the real document) that the source font is
    strictly monospace -- this filter uses that known constraint to
    reject proportional-width fonts before they can win purely on
    coincidental per-glyph correlation, which is otherwise a real risk
    (an italic proportional font scoring highest despite being
    structurally implausible was observed in testing)."""
    try:
        w_i = font.getlength("i")
        w_M = font.getlength("M")
    except AttributeError:
        w_i = font.getbbox("i")[2]
        w_M = font.getbbox("M")[2]
    if w_i == 0:
        return False
    return abs(w_i - w_M) / w_M < tolerance


def is_italic_name(path):
    """Cheap secondary filter: skip filenames that self-identify as
    italic/oblique. We know the source document is upright."""
    name = os.path.basename(path).lower()
    return "italic" in name or "oblique" in name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", required=True,
                     help="directory containing candidate .ttf/.otf files")
    ap.add_argument("--real-x", default="real_X.npy")
    ap.add_argument("--real-y", default="real_y.npy")
    ap.add_argument("--min-samples", type=int, default=3,
                     help="skip classes with fewer than this many real "
                          "samples -- too noisy to score reliably")
    ap.add_argument("--recursive", action="store_true",
                     help="search --fonts-dir recursively (e.g. point "
                          "straight at /usr/share/fonts to test every "
                          "installed font, not just a hand-picked folder)")
    ap.add_argument("--top-n", type=int, default=15,
                     help="how many top-ranked fonts to print in full "
                          "(the ranked list can get long when scanning "
                          "the whole system font tree)")
    ap.add_argument("--allow-proportional", action="store_true",
                     help="don't filter out proportional-width fonts. "
                          "Off by default because we already confirmed "
                          "by direct pixel measurement that the source "
                          "document is strictly monospace.")
    ap.add_argument("--allow-italic", action="store_true",
                     help="don't filter out italic/oblique-named fonts. "
                          "Off by default -- source document is upright.")
    args = ap.parse_args()

    real_X = np.load(args.real_x)
    real_y = np.load(args.real_y)

    # average + center-align real glyphs per class
    class_avg = {}
    for cls in sorted(set(real_y.tolist())):
        mask = real_y == cls
        if mask.sum() < args.min_samples:
            continue
        imgs = [center_of_mass_align(real_X[i]) for i in np.where(mask)[0]]
        class_avg[cls] = np.mean(imgs, axis=0)

    print(f"Scoring against {len(class_avg)} classes with >= "
          f"{args.min_samples} real samples: {sorted(class_avg.keys())}\n")

    font_paths = sorted(glob.glob(os.path.join(args.fonts_dir, "*.ttf")) +
                         glob.glob(os.path.join(args.fonts_dir, "*.otf")))
    if args.recursive:
        font_paths = sorted(
            glob.glob(os.path.join(args.fonts_dir, "**", "*.ttf"), recursive=True) +
            glob.glob(os.path.join(args.fonts_dir, "**", "*.otf"), recursive=True)
        )
    font_paths = sorted(set(font_paths))  # dedupe in case both globs overlap
    if not font_paths:
        print(f"No .ttf/.otf files found in {args.fonts_dir}")
        return

    print(f"Found {len(font_paths)} font files"
          f"{' (recursive)' if args.recursive else ''}")
    if not args.allow_italic or not args.allow_proportional:
        filters = []
        if not args.allow_proportional:
            filters.append("proportional-width")
        if not args.allow_italic:
            filters.append("italic/oblique-named")
        print(f"Filtering out {' and '.join(filters)} fonts "
              f"(use --allow-proportional / --allow-italic to disable)")
    print()

    results = []
    n_skipped_proportional, n_skipped_italic = 0, 0
    for font_path in font_paths:
        if not args.allow_italic and is_italic_name(font_path):
            n_skipped_italic += 1
            continue
        try:
            font = ImageFont.truetype(font_path, FONT_RENDER_SIZE)
        except Exception as e:
            print(f"SKIP {font_path}: couldn't load ({e})")
            continue

        if not args.allow_proportional and not is_monospace(font):
            n_skipped_proportional += 1
            continue

        scores = []
        per_class = {}
        for cls, avg_real in class_avg.items():
            rendered = render_glyph(cls, font)
            if rendered is None:
                continue
            rendered_aligned = center_of_mass_align(rendered)
            score = normalized_cross_correlation(avg_real, rendered_aligned)
            scores.append(score)
            per_class[cls] = score

        if not scores:
            print(f"SKIP {font_path}: no renderable classes")
            continue

        mean_score = float(np.mean(scores))
        results.append((mean_score, font_path, per_class))

    results.sort(reverse=True)

    print(f"Scored {len(results)} fonts "
          f"(skipped {n_skipped_proportional} proportional, "
          f"{n_skipped_italic} italic-named)")

    print("=" * 70)
    print(f"RANKED RESULTS -- top {min(args.top_n, len(results))} of "
          f"{len(results)} fonts tested (higher = closer shape match)")
    print("=" * 70)
    for rank, (score, path, per_class) in enumerate(results[:args.top_n], 1):
        print(f"{rank}. {path}  mean_score={score:.4f}")

    if results:
        best_score, best_path, best_per_class = results[0]
        print(f"\nBest match: {best_path}")
        print("Per-class scores for the winner (low outliers worth a look):")
        for cls, s in sorted(best_per_class.items(), key=lambda kv: kv[1]):
            print(f"  {cls!r:5} {s:.3f}")


if __name__ == "__main__":
    main()