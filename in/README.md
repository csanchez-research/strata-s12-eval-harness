# Inputs

This folder holds the harness inputs. `norm_stats.json` and
`config_snapshot.json` are already included in the repository; the pre-trained
+weights are published separately and must be downloaded before running the harness.


## Required file

**`encoder.pt`** — the pre-trained MAE encoder weights. Download from the
+weights release (Zenodo/HF) and save it here as `encoder.pt`.


`norm_stats.json` — the per-band normalisation statistics used at pre-training — is already present in this folder, so nothing needs to be downloaded for it.
 


The harness reads only from this folder for the encoder side; it does not import
any training code. reBEN itself is pointed to separately with `--reben-root`.

## Format

- `encoder.pt`: a state dict of the ViT-S/16 encoder (6-channel input), as
  produced by `save_encoder` in the training project. Loadable by `model_def.py`.
- `norm_stats.json`: `{"band_means": [...6...], "band_stds": [...6...], ...}` in
  the corpus band order (S2 B4, B3, B2, then S1 VV, VH, VV-VH).
