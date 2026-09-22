#!/usr/bin/env python3
"""
build_pretraining_shards.py — plan the WebDataset shards used for pre-training.

This is the exact planner that laid out the shards the STRATA-S12 encoder was
pre-trained on. It is included for reproducibility; the shards themselves are a
training-time repackaging of data already released in the corpus and are not
redistributed on their own.

Training reads differently from browsing. The stratum tree is right for
selecting a class or a season by hand, but a loader wants long sequential
reads, not two hundred thousand small ones scattered across a directory tree.
WebDataset is the usual answer: fixed-size tar files, read start to finish, one
after another.

This does not pack them. It decides what goes where and writes a companion
script (`build_shards_template.py.in`, filled in and emitted next to the lists)
that does the packing on Windows, where both disks are local.

What the plan guarantees:

*Samples are shuffled before they are split.* A shard holding one class in one
region would make the loader's shuffle buffer see nothing else for thousands of
samples, and the batches would be skewed however balanced the corpus is. The
shuffle is seeded, so the same seed lays out the same shards.

*Each sample is one prefix.* WebDataset groups files by the part of the name
before the first dot, so the pair becomes ``000042.s2.tif``, ``000042.s1.tif``
(and ``000042.json`` if sidecars are kept) and arrives as one dictionary. The
original pair_id stays in the index, so nothing is lost by renaming.

*Shards do not overlap.* Every pair appears exactly once, which is what makes an
epoch an epoch.

The pre-training subset used in the paper was built label-free (``--no-sidecars``),
seed 42, and the first 330 shards were taken (90,403 pairs).

    python build_pretraining_shards.py --manifest manifest.csv \\
        --win-source D:/strata_s12_release/dataset \\
        --win-out E:/shards --script-dir out --no-sidecars

If the output disk fills, the emitted packer stops at that shard. Point
--win-out at the next disk, re-run this planner with --start-shard at the number
it stopped on, and carry on: shards already written are not touched.

References:
    WebDataset format — https://github.com/webdataset/webdataset
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path, PureWindowsPath

# 1 GB is the usual middle of the range: large enough that the read is
# sequential and the per-file overhead disappears, small enough that a
# loader can hold several open and shuffle between them.
DEFAULT_SHARD_GB = 1.0

# Measured on the delivered corpus: 2,698.6 GB over 740,746 pairs, tiles
# only. Used to decide how many pairs go in a shard.
MB_PER_PAIR = 3.73


def win(path: str) -> str:
    return str(PureWindowsPath(*Path(path).parts))


def class_from_name(filename: str) -> str:
    """Dominant class, read from the filename rather than the sidecar.

    The name carries it already — ``S2_CROP_SUM_EUR_...`` — so reading the
    JSON of every pair to learn something the path already says would be
    hundreds of thousands of file opens for nothing.
    """
    parts = Path(filename).stem.split("_")
    return parts[1] if len(parts) > 1 else "?"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--win-source", required=True, help="the release folder as Windows sees it"
    )
    ap.add_argument(
        "--win-out", required=True, help="where the shards go, as Windows sees it"
    )
    ap.add_argument(
        "--script-dir",
        type=Path,
        required=True,
        help="where the packer script and the index are written",
    )
    ap.add_argument(
        "--win-script-dir",
        default=None,
        help="the same folder as --script-dir, as Windows sees "
        "it; defaults to the parent of --win-source",
    )
    ap.add_argument("--shard-gb", type=float, default=DEFAULT_SHARD_GB)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="strata-s12")
    ap.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="resume from this shard number, for when the previous disk filled up",
    )
    ap.add_argument(
        "--sidecars",
        action="store_true",
        default=True,
        help="include the JSON in each sample (default)",
    )
    ap.add_argument(
        "--no-sidecars",
        dest="sidecars",
        action="store_false",
        help="tiles only; smaller shards, no per-sample metadata "
        "(this is what the paper's pre-training used)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    with args.manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("ERROR: manifest is empty", file=sys.stderr)
        return 2

    # Shuffle before splitting, not after: the shard a sample lands in is
    # what the loader's buffer will see it next to, so the mixing has to
    # happen here. Seeded, so the layout is reproducible.
    random.Random(args.seed).shuffle(rows)

    per_shard = max(1, int(args.shard_gb * 1024 / MB_PER_PAIR))
    shards = [rows[i : i + per_shard] for i in range(0, len(rows), per_shard)]

    src = args.win_source.rstrip("/\\").replace("/", "\\")
    out = args.win_out.rstrip("/\\").replace("/", "\\")
    # The lists are already on a disk Windows can see: the planner writes them
    # into the release folder, which is where the script goes too. Reading
    # them there avoids copying thousands of files to where they already are.
    lists_win = (args.win_script_dir or src.rsplit("\\", 1)[0]).rstrip("/\\").replace(
        "/", "\\"
    ) + f"\\shardwork_{args.prefix}"

    print(f"manifest    : {args.manifest}")
    print(f"pairs       : {len(rows):,}")
    print(f"shard size  : {args.shard_gb:.1f} GB  (~{per_shard:,} pairs each)")
    print(f"shards      : {len(shards):,}")
    print(f"  from {args.start_shard:05d} to {len(shards) - 1:05d}")
    print(f"estimated   : {len(rows) * MB_PER_PAIR / 1024:,.0f} GB total")
    print(f"sidecars    : {'yes' if args.sidecars else 'no'}")
    print(f"seed        : {args.seed}")

    if args.dry_run:
        print("\n[DRY RUN] nothing written")
        return 0

    args.script_dir.mkdir(parents=True, exist_ok=True)
    stage = args.script_dir / f"shardwork_{args.prefix}"
    stage.mkdir(parents=True, exist_ok=True)
    for old in stage.glob("*.txt"):
        old.unlink()

    # An index of what went where, written alongside the shards. Without it,
    # tracing a training sample back to its pair_id means opening shards; with
    # it, it is a lookup.
    index_path = args.script_dir / f"{args.prefix}_shard_index.csv"
    index_fh = index_path.open("w", newline="", encoding="utf-8")
    index = csv.writer(index_fh)
    index.writerow(
        [
            "shard",
            "key",
            "pair_id",
            "tile_id",
            "class_name",
            "season",
            "region",
            "source_plan",
        ]
    )

    # One list per shard: the source path and the name the file takes inside
    # the archive, tab separated. The builder reads nothing else.
    for i, shard in enumerate(shards):
        if i < args.start_shard:
            continue
        with (stage / f"{i:05d}.txt").open("w", encoding="utf-8") as lf:
            for j, r in enumerate(shard):
                key = f"{i:05d}{j:04d}"
                for sensor in ("s2", "s1"):
                    rel = win(r[f"{sensor}_file"])
                    lf.write(f"{rel}\t{key}.{sensor}.tif\n")
                if args.sidecars:
                    rel = win(r["s2_file"])
                    lf.write(f"{Path(rel).with_suffix('.json')}\t{key}.json\n")
                index.writerow(
                    [
                        f"{args.prefix}-{i:05d}.tar",
                        key,
                        r["pair_id"],
                        r["tile_id"],
                        class_from_name(Path(r["s2_file"]).name),
                        r.get("season", ""),
                        r.get("region", ""),
                        r.get("source_plan", ""),
                    ]
                )

    # The builder runs on Windows, where both disks are local. Doing it from a
    # Linux container over a network mount was measured here at ten to a
    # hundred times slower, so the packing is emitted as a script to run natively.
    template = Path(__file__).resolve().parent / "build_shards_template.py.in"
    if not template.exists():
        print(f"ERROR: template not found: {template}", file=sys.stderr)
        index_fh.close()
        return 2
    body = template.read_text(encoding="utf-8")
    for key, value in (
        ("{SRC}", src),
        ("{OUT}", out),
        ("{LISTS}", lists_win),
        ("{VERIFIED}", out + "\\_verified"),
        ("{MIN_FREE_GB}", "3"),
        ("{PREFIX}", args.prefix),
    ):
        body = body.replace(key, value)

    index_fh.close()
    packer = args.script_dir / f"build_shards_{args.prefix}.py"
    packer.write_text(body, encoding="utf-8")

    print()
    print(f"  lists : {stage}")
    print(f"  index : {index_path}")
    print(f"  script: {packer}")
    print()
    print("Run from Windows:")
    print(f"  python {packer.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
