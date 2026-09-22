"""
model_def.py - self-contained definition of the ViT-S/16 encoder (6 channels)
for the probe. It imports nothing from the training project: it carries its own
minimal copy of the encoder architecture, so the harness is publishable on its
own.

ENCODER only (not the MAE decoder): the probe freezes the encoder and extracts
features. It loads the weights saved by `save_encoder` during training
 (`{"encoder": state_dict, "arch": ..., "in_chans": 6}`).

Feature used by the probe: the CLS token after the encoder (dim 384).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


def _posenc_2d_sincos(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sin/cos positional encoding (same as in pre-training)."""
    assert embed_dim % 4 == 0
    g = np.arange(grid_size, dtype=np.float32)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    dim_q = embed_dim // 4
    omega = 1.0 / (10000 ** (np.arange(dim_q, dtype=np.float32) / dim_q))

    def _emb(pos):
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    pos = np.concatenate([_emb(gy), _emb(gx)], axis=1)
    return torch.from_numpy(pos).float().unsqueeze(0)


class ViTEncoder(nn.Module):
    """6-channel ViT-Small/16 encoder. Returns the CLS token as feature."""

    ENC_DIM = 384
    ENC_DEPTH = 12
    ENC_HEADS = 6

    def __init__(self, img_size: int = 224, patch: int = 16, in_chans: int = 6):
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.patch_embed = nn.Conv2d(in_chans, self.ENC_DIM,
                                     kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.ENC_DIM))
        self.register_buffer(
            "pos_embed",
            torch.cat([
                torch.zeros(1, 1, self.ENC_DIM),
                _posenc_2d_sincos(self.ENC_DIM, self.grid),
            ], dim=1),
        )
        self.blocks = nn.ModuleList([
            Block(self.ENC_DIM, self.ENC_HEADS, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(self.ENC_DEPTH)
        ])
        self.norm = nn.LayerNorm(self.ENC_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 6, 224, 224) -> (B, 384) CLS token. NO masking (the probe sees all)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, E)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


def load_encoder(path: str | Path, device: str = "cpu") -> ViTEncoder:
    """Load the encoder from an encoder.pt saved by save_encoder.

    The training project uses a timm-style PatchEmbed and stores the projection
    as `patch_embed.proj.*`. Here `patch_embed` is a plain Conv2d, so its keys
    are `patch_embed.*`. They are remapped explicitly.

    Hard-fails if any parameter is left unloaded: a half-loaded encoder is
    indistinguishable from a pre-trained one at a glance and would falsify the
    probe. Better to abort than to measure noise.

    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    in_chans = ckpt.get("in_chans", 6)
    model = ViTEncoder(in_chans=in_chans)
    state = ckpt.get("encoder", ckpt)

    # remap training -> probe (timm PatchEmbed -> plain Conv2d)
    n_remap = sum(1 for k in state if "patch_embed.proj." in k)
    state = {k.replace("patch_embed.proj.", "patch_embed."): v
             for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    # pos_embed is fixed sincos (buffer), not a learned weight: tolerated
    missing = [k for k in missing if not k.endswith("pos_embed")]
    unexpected = [k for k in unexpected if not k.endswith("pos_embed")]

    if missing or unexpected:
        raise RuntimeError(
            "[model_def] the encoder was NOT fully loaded; aborting.\n"
            f"  unloaded (would stay RANDOM): {missing}\n"
            f"  unexpected in checkpoint    : {unexpected}\n"
            "Fix the key mapping before running the probe."

        )

    print(f"  [model_def] encoder loaded: {len(state)} tensors"
          f"{f' ({n_remap} remapped)' if n_remap else ''}, 0 unloaded")

    model.eval()
    return model


def build_random_encoder(in_chans: int = 6) -> ViTEncoder:
    """Encoder with random weights: the probe baseline (not pre-trained)."""
    model = ViTEncoder(in_chans=in_chans)
    model.eval()
    return model


if __name__ == "__main__":
    # smoke: build a random encoder and pass a tensor
    m = build_random_encoder()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(2, 6, 224, 224)
    with torch.no_grad():
        f = m(x)
    print(f"encoder params: {n:.1f}M")
    print(f"feature out   : {tuple(f.shape)}  (expected (2, 384))")
