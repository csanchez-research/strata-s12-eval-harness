"""
extract_features.py - passes the reBEN patches through the FROZEN encoder and
caches the feature vectors (384-dim) to disk, so the linear probe can reuse them
without recomputing.

Features are extracted from two encoders:
  - the PRE-TRAINED one (in/encoder.pt): the representation under evaluation.
  - a RANDOM-INIT one (same architecture, untrained): the paper's baseline.

Features go to features/<split>_<which>.npz together with the 19-hot labels, so
the probe only has to load matrices and train the classifier.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_def import build_random_encoder, load_encoder
from reben_loader import ReBENDataset


@torch.no_grad()
def extract(encoder, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N,384], labels [N,19]) for the split."""
    encoder.eval().to(device)
    feats, labels = [], []
    for xs, ys in loader:
        xs = xs.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            f = encoder(xs)                    # (B, 384)
        feats.append(f.float().cpu().numpy())
        labels.append(ys.numpy())
        del xs, f                              # free the batch VRAM
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reben-root", required=True,
                    help="root of the extracted reBEN (contains BigEarthNet-S1/S2)")
    ap.add_argument("--metadata", required=True, help="metadata.parquet")
    ap.add_argument("--encoder", default="in/encoder.pt")
    ap.add_argument("--stats", default="in/norm_stats.json")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--which", nargs="+", default=["pretrained", "random"],
                    help="which encoders to extract: pretrained and/or random (baseline)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--max-patches", type=int, default=None,
                    help="limit patches per split (quick tests)")
    ap.add_argument("--max-train", type=int, default=None,
                    help="limit ONLY the train split (test stays complete, so "
                         "that the results table covers the whole benchmar")
    ap.add_argument("--seed", type=int, default=0,
                    hhelp="random baseline seed (for reproducibility)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {dev}  ·  splits: {args.splits}  ·  which: {args.which}")

    # build the encoders once
    encoders = {}
    if "pretrained" in args.which:
        encoders["pretrained"] = load_encoder(args.encoder, device=dev)
        print(f"pre-trained encoder loaded from {args.encoder}")
    if "random" in args.which:
        torch.manual_seed(args.seed)
        encoders["random"] = build_random_encoder()
        print(f"random-init encoder created (seed {args.seed})")

    for split in args.splits:
       # test/validation go complete; only train allows a cap.
        cap = args.max_patches
        if args.max_train is not None and split == "train":
            cap = args.max_train
        ds = ReBENDataset(args.reben_root, args.metadata, args.stats,
                          split=split, max_patches=cap)
        loader = DataLoader(ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=False)
        print(f"\n[{split}] {len(ds)} patches")
        for which, enc in encoders.items():
            t0 = __import__("time").time()
            feats, labels = extract(enc, loader, dev)
            dt = __import__("time").time() - t0
            path = out / f"{split}_{which}.npz"
            np.savez_compressed(path, features=feats, labels=labels)
            print(f"  {which:10s}: {feats.shape} -> {path}  ({dt:.0f}s)")

    print("\nExtraction complete. The probe will use these .npz files.")


if __name__ == "__main__":
    main()
