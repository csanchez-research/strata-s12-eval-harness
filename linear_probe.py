"""
linear_probe.py - trains a LINEAR classifier on the frozen features and measures
micro-mAP (primary) and macro-mAP (descriptor) on the reBEN test split.

This is the paper's "frozen linear probe": the encoder is untouched; only a
linear layer (384 -> 19) is trained on the features cached by extract_features.
It is run for the PRE-TRAINED encoder and for the random BASELINE, over several
seeds, so aggregate.py can apply the non-overlap criterion.

Multi-label: 19 sigmoid outputs, BCE loss. The metric is per-class average
precision, averaged (micro and macro), standard on BigEarthNet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


def _load_split(feat_dir: Path, split: str, which: str):
    d = np.load(feat_dir / f"{split}_{which}.npz")
    return d["features"].astype(np.float32), d["labels"].astype(np.float32)


def train_linear_probe(
    Xtr, Ytr, Xte, Yte, seed: int, epochs: int = 100, lr: float = 1e-3,
    weight_decay: float = 0.0, device: str = "cpu",
) -> dict:
    """Trains the linear layer and returns micro/macro-mAP on test."""
    torch.manual_seed(seed)
    np.random.seed(seed)   # noqa: NPY002 - intentional global seeding

    in_dim, n_cls = Xtr.shape[1], Ytr.shape[1]
    clf = nn.Linear(in_dim, n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    xtr = torch.from_numpy(Xtr).to(device)
    ytr = torch.from_numpy(Ytr).to(device)

    clf.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(clf(xtr), ytr)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(torch.from_numpy(Xte).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    # micro pools all labels: globally there are always positives.
    micro = average_precision_score(Yte, probs, average="micro")

    # macro: average ONLY over classes with >=1 positive in test. A class with
    # no positives gives an undefined AP (the UserWarning) and contaminates the
    # average. Filtering is the standard on BigEarthNet.

    valid = Yte.sum(axis=0) > 0                      # (C,) bool
    n_valid = int(valid.sum())
    per_class_ap: dict[int, float] = {}
    if n_valid == 0:
        macro = float("nan")
    else:
        ap_per_class = average_precision_score(
            Yte[:, valid], probs[:, valid], average=None
        )
        macro = float(np.mean(ap_per_class))
        # map each AP to its REAL class index (0-18), not the filtered order
        valid_idx = np.where(valid)[0]
        per_class_ap = {int(c): float(ap)
                        for c, ap in zip(valid_idx, np.atleast_1d(ap_per_class),
                                         strict=True)}

    return {
        "micro_mAP": float(micro),
        "macro_mAP": float(macro),
        "n_classes_valid": n_valid,
        "n_classes_total": int(Yte.shape[1]),
        "per_class_ap": per_class_ap,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="features")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--which", nargs="+", default=["pretrained", "random"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--require-classes", type=int, default=0,
                    help="Abort if any test split has fewer than N classes "
                         "with positives. Set to 19 for the final run; leave it "
                         "at 0 for smoke tests.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="out/probe_results.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = Path(args.feat_dir)
    results: dict[str, list[dict]] = {w: [] for w in args.which}

    for which in args.which:
        Xtr, Ytr = _load_split(feat_dir, args.train_split, which)
        Xte, Yte = _load_split(feat_dir, args.test_split, which)
        print(f"\n[{which}] train {Xtr.shape}  test {Xte.shape}")

        n_valid_test = int((Yte.sum(axis=0) > 0).sum())
        if args.require_classes and n_valid_test < args.require_classes:
            raise SystemExit(
                f"[{which}] test only has {n_valid_test} classes with positives, "
                f"{args.require_classes} are required. macro-mAP over a subset is "
                f"NOT comparable with the 19-class benchmark. Use the complete "
                f"test split (more patches)."

            )

        for seed in range(args.seeds):
            r = train_linear_probe(Xtr, Ytr, Xte, Yte, seed=seed,
                                   epochs=args.epochs, lr=args.lr, device=dev)
            results[which].append(r)
            print(f"  seed {seed}: micro-mAP {r['micro_mAP']:.4f}  "
                  f"macro-mAP {r['macro_mAP']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-seed results -> {out}")
    print("Run aggregate.py for the results table and the criterion.")



if __name__ == "__main__":
    main()
