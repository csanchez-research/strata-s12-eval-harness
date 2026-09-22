#!/usr/bin/env python
"""
run_probe.py - orchestrates the full harness end to end.

Stages:
  features : extracts frozen reBEN features (pre-trained encoder + random)
  probe    : trains the linear probe on the features, N seeds
  aggregate: applies the criterion, generates the table and the figure


Runs everything, or a single stage with --stage.
 

Typical use (with inputs in in/ and reBEN extracted):
     python run_probe.py --reben-root /reben --metadata /reben/metadata.parquet
     python run_probe.py --reben-root /reben --metadata /reben/metadata.parquet \

        --max-patches 500        # quick test
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    ret = subprocess.run([sys.executable, *cmd]).returncode
    if ret != 0:
        raise SystemExit(f"stage failed (code {ret}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reben-root", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--encoder", default="in/encoder.pt")
    ap.add_argument("--stats", default="in/norm_stats.json")
    ap.add_argument("--stage", choices=["all", "features", "probe", "aggregate"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--require-classes", type=int, default=0,
                    help="Abort if test has fewer than N classes with "
                         "positives. Set to 19 for the final run; 0 for smoke.")
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None,
                    help="limit ONLY train; test stays complete (results table).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    if args.stage in ("all", "features"):
        cmd = ["extract_features.py",
               "--reben-root", args.reben_root, "--metadata", args.metadata,
               "--encoder", args.encoder, "--stats", args.stats,
               "--batch-size", str(args.batch_size),
               "--num-workers", str(args.num_workers)]
        if args.max_patches:
            cmd += ["--max-patches", str(args.max_patches)]
        if args.max_train:
            cmd += ["--max-train", str(args.max_train)]
        run(cmd)

    if args.stage in ("all", "probe"):
        probe_cmd = ["linear_probe.py", "--seeds", str(args.seeds)]
        if args.require_classes:
            probe_cmd += ["--require-classes", str(args.require_classes)]
        run(probe_cmd)

    if args.stage in ("all", "aggregate"):
        run(["aggregate.py"])

    print("\n=== HARNESS COMPLETE ===")
    print("Results in out/: probe_table.tex, probe_figure.pdf, summary.json")
 



if __name__ == "__main__":
    main()
