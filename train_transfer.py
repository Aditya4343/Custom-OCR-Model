"""
Transfer-learning trainer for the custom character classifier.

Backbone: torchvision's MobileNetV3-Small, pretrained on ImageNet. Almost
all of it stays FROZEN -- we're reusing its general-purpose edge/stroke/
curve features, not asking it to learn our font from scratch. Only a new
small classification head (and optionally the last block) gets trained.
This is what makes it CPU-feasible: the expensive part of the network
never needs a gradient computed through it in "frozen backbone" mode.

Two-phase training, controlled by --unfreeze-last-block / --unfreeze-all:
  Phase 1 (always): train only the new head. Fast, very CPU-friendly,
      a good first checkpoint to sanity-check the whole pipeline runs.
  Phase 2a (--unfreeze-last-block): additionally unfreeze the backbone's
      last conv block. Modest extra cost, workable on CPU.
  Phase 2b (--unfreeze-all, GPU/MPS only): full fine-tune, every layer
      trainable, two learning rates (fresh head higher, pretrained
      backbone much lower to avoid catastrophic forgetting). This is
      the "fully customized" option -- with a GPU or Apple Silicon MPS
      backend and a dataset this small, a full fine-tune finishes in
      minutes, not hours, and gives every layer room to adapt to this
      specific font rather than reusing generic ImageNet features
      unchanged. Auto-detects CUDA, then MPS (Apple Silicon), then
      falls back to CPU.

Evaluation is the important part, and matches the methodology used
earlier: real harvested glyphs are held out in a way that NEVER lets an
original real glyph (or any of its augmented copies) leak between train
and validation, and only real (never synthetic) data is used to report
the number you should actually trust. Synthetic data can make training
loss look great while telling you nothing about real-world accuracy --
the real held-out set is the only honest signal.

Usage:
    pip install torch torchvision --break-system-packages   # CPU build
    python3 synth_data.py --out data/ --per-class 300
    python3 train_transfer.py --data data/ --epochs 15
    python3 train_transfer.py --data data/ --epochs 10 --unfreeze-last-block
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class GlyphDataset(Dataset):
    def __init__(self, X, y, class_to_idx):
        # MobileNetV3 expects 3-channel input; replicate the single
        # grayscale channel rather than retraining the stem from scratch.
        self.X = X
        self.y = np.array([class_to_idx[c] for c in y])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        t = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)
        return t, self.y[i]


def build_model(num_classes, unfreeze_last_block=False, unfreeze_all=False):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)

    if unfreeze_all:
        # GPU/MPS-only: full fine-tune, every layer trainable. Needs
        # meaningfully more data/compute than head-only or last-block
        # training to avoid catastrophic forgetting, but with GPU/MPS
        # access it's cheap enough to just do given how small this model
        # and dataset are.
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            for p in model.features[-1].parameters():
                p.requires_grad = True

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, idx_to_class=None):
    model.eval()
    correct, total = 0, 0
    per_class_correct, per_class_total = {}, {}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if idx_to_class is not None:
                for p, t in zip(pred.tolist(), y.tolist()):
                    per_class_total[t] = per_class_total.get(t, 0) + 1
                    if p == t:
                        per_class_correct[t] = per_class_correct.get(t, 0) + 1
    acc = correct / max(total, 1)
    if idx_to_class is not None:
        print("\nPer-class held-out accuracy (class: correct/total):")
        for idx in sorted(per_class_total, key=lambda i: -per_class_total[i]):
            c = idx_to_class[idx]
            n_correct = per_class_correct.get(idx, 0)
            n_total = per_class_total[idx]
            print(f"  {c!r:5} {n_correct}/{n_total}  ({n_correct/n_total:.0%})")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--unfreeze-last-block", action="store_true")
    ap.add_argument("--unfreeze-all", action="store_true",
                     help="Full fine-tune -- every layer trainable. Only "
                          "worth it with real GPU/MPS access; on CPU this "
                          "will be slow enough to defeat the point.")
    ap.add_argument("--lr-full", type=float, default=1e-4,
                     help="LR for all backbone params when --unfreeze-all "
                          "is set (kept lower than --lr-head so pretrained "
                          "weights adapt rather than get overwritten)")
    ap.add_argument("--real-glyphs", default=None,
                     help="path to real real_X.npy (digits+letters combined; "
                          "for the held-out eval set); defaults to "
                          "real_X.npy, falling back to digit_X.npy")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    X = np.load(f"{args.data}/X.npy")
    y = np.load(f"{args.data}/y.npy")
    src = np.load(f"{args.data}/src.npy")

    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} classes: {classes}")

    # ---- Honest held-out evaluation set ----
    # Pull a real, ground-truth-labeled, never-trained-on slice directly
    # from the ORIGINAL harvested glyphs (digits + letters combined, not
    # the augmented copies mixed into data/X.npy), so training and
    # evaluation touch completely disjoint pixels, not just disjoint
    # "logical" samples.
    real_glyph_path = "real_X.npy" if __import__("os").path.exists("real_X.npy") else "digit_X.npy"
    real_label_path = "real_y.npy" if __import__("os").path.exists("real_y.npy") else "digit_y.npy"
    try:
        real_X_orig = np.load(real_glyph_path)
        real_y_orig = np.load(real_label_path)
    except FileNotFoundError:
        real_X_orig, real_y_orig = None, None

    if real_X_orig is not None:
        rng = np.random.default_rng(0)
        n = len(real_X_orig)
        holdout_frac = 0.3
        idx = rng.permutation(n)
        n_holdout = max(1, int(n * holdout_frac))
        holdout_idx = set(idx[:n_holdout].tolist())
        print(f"Held out {n_holdout}/{n} ORIGINAL real glyphs for eval "
              f"(and excluding all their augmented copies from training)")
    else:
        holdout_idx = set()
        print("WARNING: no real_X.npy/digit_X.npy found -- can't build a real-data "
              "held-out set. Accuracy reported below will be on synthetic "
              "data only and should NOT be trusted as a real-world number.")

    # data/X.npy's real-derived rows came from load_real_glyphs() iterated
    # in original order, oversampled by a fixed factor each -- reconstruct
    # which rows correspond to which original index to exclude holdouts.
    train_mask = np.ones(len(X), dtype=bool)
    if real_X_orig is not None:
        real_rows = np.where(src == "real")[0]
        oversample = len(real_rows) // len(real_X_orig) if len(real_X_orig) else 0
        for j, row_i in enumerate(real_rows):
            orig_idx = j // max(oversample, 1)
            if orig_idx in holdout_idx:
                train_mask[row_i] = False

    X_train, y_train = X[train_mask], y[train_mask]
    print(f"Training on {len(X_train)} samples "
          f"({(src[train_mask]=='synthetic').sum()} synthetic, "
          f"{(src[train_mask]=='real').sum()} real-derived)")

    train_ds = GlyphDataset(X_train, y_train, class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = build_model(len(classes), unfreeze_last_block=args.unfreeze_last_block,
                         unfreeze_all=args.unfreeze_all)
    model.to(device)

    if args.unfreeze_all:
        # Two LR groups: fresh head trains faster/higher LR, pretrained
        # backbone gets a much lower LR so it adapts rather than
        # overwrites what ImageNet pretraining already learned
        # (catastrophic forgetting risk otherwise).
        backbone_params = [p for n, p in model.named_parameters()
                            if not n.startswith("classifier")]
        head_params = list(model.classifier.parameters())
        param_groups = [
            {"params": head_params, "lr": args.lr_head},
            {"params": backbone_params, "lr": args.lr_full},
        ]
    else:
        param_groups = [{"params": model.classifier.parameters(), "lr": args.lr_head}]
        if args.unfreeze_last_block:
            param_groups.append({"params": model.features[-1].parameters(),
                                  "lr": args.lr_backbone})
    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/n_batches:.4f}")

    if real_X_orig is not None and holdout_idx:
        holdout_X = np.array([real_X_orig[i] for i in holdout_idx])
        holdout_y = np.array([str(real_y_orig[i]) for i in holdout_idx])
        # resize to match training image size (32x32) the same way
        # synth_data.py does, for a fair apples-to-apples comparison
        import cv2
        holdout_X_resized = np.array([
            cv2.resize(img.astype(np.float32), (32, 32)) for img in holdout_X
        ])
        holdout_ds = GlyphDataset(holdout_X_resized, holdout_y, class_to_idx)
        holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        acc = evaluate(model, holdout_loader, device, idx_to_class=idx_to_class)
        print(f"\n=== HONEST HELD-OUT ACCURACY on {len(holdout_X)} real, "
              f"never-trained-on glyphs: {acc:.3f} ===")
        print("This is the number that matters -- everything else "
              "(training loss, synthetic-data accuracy) can look good "
              "while this stays low if the synthetic font doesn't match "
              "well enough. Compare against the from-scratch baseline "
              "(~0.56-0.57) to see whether transfer learning actually "
              "helped here.")

    torch.save(model.state_dict(), "glyph_classifier.pt")
    import json
    with open("glyph_classifier_classes.json", "w") as f:
        json.dump(classes, f)  # classes[i] == character for output index i
    print("Saved model to glyph_classifier.pt "
          "and class mapping to glyph_classifier_classes.json")


if __name__ == "__main__":
    main()