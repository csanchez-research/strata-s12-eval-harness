"""
reben_loader.py - reads a reBEN patch (BigEarthNet v2.0) and turns it into the
SAME 6-channel stack the encoder was pre-trained on, so the probe is evaluated
with the correct input distribution.

reBEN layout (from the Dataset Description):
  BigEarthNet-S2/<tile>/<patch_id>/<patch_id>_B04.tif  (red)
                                   <patch_id>_B03.tif  (green)
                                   <patch_id>_B02.tif  (blue)   ... and B01..B12
   BigEarthNet-S1/<tile>/<s1_name>/<s1_name>_VV.tif
                                   <s1_name>_VH.tif


The 6 channels, in corpus ORDER: [S2_B4, S2_B3, S2_B2, S1_VV, S1_VH, S1_VV-VH].
The sixth (VV-VH) is derived, as in the corpus. It is normalised with the same
per-band statistics from pre-training (norm_stats.json).


The patch<->s1 pairing and the 19-class labels come from metadata.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# reBEN's 19 classes in fixed order (alphabetical, as they appear in the parquet).
# Fixed here so that the one-hot vector is reproducible.

REBEN_19_CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(REBEN_19_CLASSES)}


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)  # (H, W)


def _s2_tile_of(patch_id: str) -> str:
    """S2 tile = patch_id without row and column (the last 2 components).

    'S2A_..._T33UUP_26_57' -> 'S2A_..._T33UUP'  (the S2 tile includes the zone)
    """
    return patch_id.rsplit("_", 2)[0]


def _s1_tile_of(s1_name: str) -> str:
    """S1 tile = s1_name without zone, row and column (the last 3).

    'S1B_..._20170612T165809_33UUP_26_57' -> 'S1B_..._20170612T165809'
    (the S1 tile does NOT include the zone '33UUP', hence 3 and not 2.)
    """
    return s1_name.rsplit("_", 3)[0]


class ReBENDataset(Dataset):
    """reBEN patches as (normalised 6x120x120 tensor, 19-hot label).

    The patch list and their labels come from metadata.parquet; the paths are
    derived deterministically from patch_id / s1_name (without walking the
    tree). reBEN patches are 120x120; the encoder expects 224x224, so they are
    resized (bilinear), as in the literature.

    """

    def __init__(
        self,
        reben_root: str | Path,
        metadata_parquet: str | Path,
        norm_stats_json: str | Path,
        split: str,                       # 'train' | 'validation' | 'test'
        img_size: int = 224,
        max_patches: int | None = None,   # for quick tests
    ):
        self.root = Path(reben_root)
        self.s2_root = self.root / "BigEarthNet-S2"
        self.s1_root = self.root / "BigEarthNet-S1"
        self.img_size = img_size

        df = pd.read_parquet(metadata_parquet)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_patches and max_patches < len(df):
             # shuffle with a FIXED seed before truncating: reproducible subset
            # (same seed -> same patches) but representative of the whole split,
            # not of the first geographic region in the parquet.

            df = df.sample(n=max_patches, random_state=42).reset_index(drop=True)
        self.df = df

        with open(norm_stats_json) as f:
            s = json.load(f)
        self.means = torch.tensor(s["band_means"], dtype=torch.float32).view(-1, 1, 1)
        self.stds = torch.clamp(
            torch.tensor(s["band_stds"], dtype=torch.float32).view(-1, 1, 1), min=1e-6
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_6ch(self, row) -> np.ndarray:
        patch_id = row["patch_id"]
        s1_name = row["s1_name"]
        s2_dir = self.s2_root / _s2_tile_of(patch_id) / patch_id
        s1_dir = self.s1_root / _s1_tile_of(s1_name) / s1_name

        b4 = _read_band(s2_dir / f"{patch_id}_B04.tif")
        b3 = _read_band(s2_dir / f"{patch_id}_B03.tif")
        b2 = _read_band(s2_dir / f"{patch_id}_B02.tif")
        vv = _read_band(s1_dir / f"{s1_name}_VV.tif")
        vh = _read_band(s1_dir / f"{s1_name}_VH.tif")
        vv_vh = vv - vh  # sixth derived band, as in the corpus
        return np.stack([b4, b3, b2, vv, vh, vv_vh], axis=0)  # (6, H, W)

    def _labels_19hot(self, row) -> torch.Tensor:
        vec = torch.zeros(len(REBEN_19_CLASSES), dtype=torch.float32)
        for lab in row["labels"]:
            idx = CLASS_TO_IDX.get(lab)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        arr = self._load_6ch(row)                      # (6, H, W)
        x = torch.from_numpy(arr)
        # resize to img_size (reBEN 120 -> 224), bilinear
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        x = (x - self.means) / self.stds               # normalise per band
        y = self._labels_19hot(row)
        return x, y
