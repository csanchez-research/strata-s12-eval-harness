# Pre-training shards

The STRATA-S12 encoder was pre-trained on the first **330 seed-shuffled
WebDataset shards** (**90,403 pairs**) drawn from the STRATA-S12 corpus. These
shards are a training-time repackaging of data already released in the corpus,
so they are **not redistributed separately**. This folder contains the exact
code that produced them, for reproducibility.

## Recipe

- **Seed:** 42 (samples are shuffled *before* they are split into shards, so any
  prefix is a representative draw across all class × season × region strata).
- **Shard size:** ~1 GB, ≈274 pairs per shard.
- **Label-free:** built with `--no-sidecars` — tiles only, no per-sample JSON.
  Self-supervised pre-training must not see labels, and the sequential keys
  (`000042.s2.tif`, `000042.s1.tif`) carry no class information.
- **Subset:** the first 330 shards were used.

## Files

- `build_pretraining_shards.py` — the planner. Reads the corpus manifest and the
  release tree, seed-shuffles, splits into shards, writes the per-shard file
  lists, a `*_shard_index.csv` (maps each shard key back to its `pair_id`), and
  emits the packer from the template below.
- `build_shards_template.py.in` — the packer template the planner fills in and
  writes out. It streams each file straight into a `.tar` (no copy, link or
  rename on disk) and can deep-validate every tile with `--validate`.

## Running it

```bash
# 1) plan (writes the lists, the index, and the packer)
python build_pretraining_shards.py \
    --manifest /path/to/manifest.csv \
    --win-source D:/strata_s12_release/dataset \
    --win-out    E:/shards \
    --script-dir out \
    --no-sidecars

# 2) pack (run the emitted script natively on Windows)
python out/build_shards_strata-s12.py
```

## Reproducibility note

The recipe reproduces a representative subset equivalent to the one used for
pre-training. Reproducing the shards **byte for byte** additionally requires the
corpus manifest in its exact original row order and the same flags; the paper's
claim ("any prefix is a representative draw") holds regardless of that exact
ordering.
