"""
Real end-to-end test: run the trained classifier on ACTUAL, uncut cell
crops -- not pre-segmented, ground-truth-sliced glyphs like every prior
accuracy number in this project. This is what the OCR pipeline would
actually have to do: given a cell image and NO known answer, find the
characters and read them.

Critically different from harvest_*.py: those scripts knew the correct
text in advance and used it to slice cells into exactly the right number
of pieces. Real inference doesn't have that luxury -- it has to segment
blind. This uses:
  1. Word-gap splitting (same technique as harvest_letters.py) to find
     word boundaries via wide ink gaps.
  2. FIXED-PITCH character slicing within each word -- divide word width
     by the font's known monospace pitch (~10.2-10.3px, measured earlier
     against this document) to get a character count, then equal-slice.
     This is different from harvest_letters.py, which divided by the
     KNOWN character count of the known word. Here the count itself is
     inferred from geometry, exactly as a real deployment would have to.

Usage:
    python3 infer_test.py --crop cell_r25_c2.png              # single glyph
    python3 infer_test.py --row-text-test                     # whole rows
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.models import mobilenet_v3_small

CHAR_PITCH = 10.25  # measured earlier: 23 chars over 237px on this document
INK_THRESHOLD = 150
WORD_GAP_MIN_PX = 5
TARGET = 22


def load_model():
    with open("glyph_classifier_classes.json") as f:
        classes = json.load(f)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    state = torch.load("glyph_classifier.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, classes


def tight_crop(bin_img):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        return None
    return bin_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_model_input(glyph_bin):
    img = Image.fromarray((glyph_bin * 255).astype(np.uint8)).resize(
        (TARGET, TARGET), Image.LANCZOS)
    arr = np.array(img) / 255.0
    t = torch.from_numpy(arr).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    return t


def split_into_words(bw_row):
    col_ink = bw_row.sum(axis=0)
    has_ink = col_ink > 1
    words = []
    in_word = False
    start = 0
    gap_run = 0
    for i, v in enumerate(has_ink):
        if v:
            if not in_word:
                start = i
                in_word = True
            gap_run = 0
        else:
            if in_word:
                gap_run += 1
                if gap_run >= WORD_GAP_MIN_PX:
                    words.append((start, i - gap_run + 1))
                    in_word = False
    if in_word:
        words.append((start, len(has_ink)))
    return words


@torch.no_grad()
def classify_glyph(model, classes, glyph_bin):
    x = to_model_input(glyph_bin)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    return classes[idx], probs[idx].item()


@torch.no_grad()
def read_cell(model, classes, gray_crop, verbose=False):
    """Full blind pipeline: word-split, then fixed-pitch char-split within
    each word, then classify each piece. No ground truth used anywhere."""
    bw = (gray_crop < INK_THRESHOLD).astype(np.uint8)
    words_px = split_into_words(bw)
    result_words = []
    for x0, x1 in words_px:
        word_bw = bw[:, x0:x1]
        tc = tight_crop(word_bw)
        if tc is None:
            continue
        w = tc.shape[1]
        n_chars = max(1, round(w / CHAR_PITCH))
        edges = np.linspace(0, w, n_chars + 1).astype(int)
        chars = []
        for i in range(n_chars):
            piece = tc[:, edges[i]:edges[i + 1]]
            pt = tight_crop(piece)
            if pt is None or pt.shape[0] < 3 or pt.shape[1] < 2:
                continue
            ch, conf = classify_glyph(model, classes, pt)
            chars.append(ch)
            if verbose:
                print(f"    glyph -> {ch!r} (confidence {conf:.2f})")
        result_words.append("".join(chars))
    return " ".join(result_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", help="path to a single cell crop image "
                                    "(e.g. cell_r25_c2.png)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model, classes = load_model()
    print(f"Loaded model, {len(classes)} classes")

    if args.crop:
        im = Image.open(args.crop).convert("L")
        arr = np.array(im)
        text = read_cell(model, classes, arr, verbose=args.verbose)
        print(f"\n{args.crop} -> predicted: {text!r}")
    else:
        print("Pass --crop <path> to test a single cell crop image, e.g.:")
        print("  python3 infer_test.py --crop cell_r25_c2.png --verbose")
        print("  python3 infer_test.py --crop cell_r26_c2.png --verbose")


if __name__ == "__main__":
    main()
