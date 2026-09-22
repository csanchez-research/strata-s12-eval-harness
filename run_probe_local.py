#!/usr/bin/env python
"""
run_probe_local.py - full probe in ONE command, reading from local disk (/tmp)
like training did (corpus_feeder), not live over 9p.

NOTE: this is the authors' environment-specific wrapper (fixed paths such as
/workspace and /reben). The portable entry point is run_probe.py.

Does everything by itself:
  train: copies 40k to /tmp -> extracts -> deletes /tmp
  test:  copies the full test set to /tmp -> extracts -> deletes /tmp
  probe + aggregate -> table

Usage:
     python run_probe_local.py

"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

S2_BANDS = ["B04", "B03", "B02"]
S1_BANDS = ["VV", "VH"]

def _files(row):
    pid, s1 = row["patch_id"], row["s1_name"]
    s2 = Path("BigEarthNet-S2") / pid.rsplit("_", 2)[0] / pid
    s1d = Path("BigEarthNet-S1") / s1.rsplit("_", 3)[0] / s1
    return [s2 / f"{pid}_{b}.tif" for b in S2_BANDS] + \
           [s1d / f"{s1}_{b}.tif" for b in S1_BANDS]

def copy_and_extract(split, max_patches, src, tmp, probe, seed, workers, batch):
    df = pd.read_parquet(src / "metadata.parquet")
    df = df[df["split"] == split]
    if max_patches and max_patches < len(df):
        df = df.sample(n=max_patches, random_state=seed)
    df = df.reset_index(drop=True)
    print(f"\n=== {split}: {len(df)} patches ===")

    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copyfile(src / "metadata.parquet", tmp / "metadata.parquet")

    missing = 0
    t0 = time.time()
    for row in tqdm(list(df.itertuples()), desc=f"copy {split}", unit="patch"):
        for frel in _files(row._asdict()):
            fs, fd = src / frel, tmp / frel
            if not fs.exists():
                missing += 1; continue
            fd.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fs, fd)
    print(f"copied in {time.time()-t0:.0f}s" + (f"  ({missing} not found!)" if missing else ""))
    if missing:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(f"{missing} .tif not found; aborting.")

    cmd = [sys.executable, "extract_features.py",
           "--reben-root", str(tmp), "--metadata", str(tmp / "metadata.parquet"),
           "--splits", split, "--num-workers", str(workers), "--batch-size", str(batch)]
    if split == "train" and max_patches:
        cmd += ["--max-train", str(max_patches)]
    print(f">>> extract {split} from /tmp")
    ret = subprocess.run(cmd, cwd=probe).returncode
    shutil.rmtree(tmp, ignore_errors=True)
    if ret != 0:
        raise SystemExit(f"extract {split} failed (code {ret})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/reben")
    ap.add_argument("--tmp", default="/tmp/reben")
    ap.add_argument("--probe-dir", default="/workspace/probe")
    ap.add_argument("--max-train", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    src, tmp, probe = Path(args.src), Path(args.tmp), Path(args.probe_dir)

    # clean start
    feat = probe / "features"
    if feat.exists():
        shutil.rmtree(feat)

    copy_and_extract("train", args.max_train, src, tmp, probe, args.seed, args.num_workers, args.batch_size)
    copy_and_extract("test", None, src, tmp, probe, args.seed, args.num_workers, args.batch_size)

    print("\n=== probe + aggregate ===")
    r = subprocess.run([sys.executable, "linear_probe.py", "--seeds", str(args.seeds),
                        "--require-classes", "19"], cwd=probe).returncode
    if r != 0:
        raise SystemExit(f"linear_probe failed ({r})")
    r = subprocess.run([sys.executable, "aggregate.py"], cwd=probe).returncode
    if r != 0:
        raise SystemExit(f"aggregate failed ({r})")
    print(f"\n=== COMPLETE. Results in {probe/'out'} ===")


if __name__ == "__main__":
    main()
