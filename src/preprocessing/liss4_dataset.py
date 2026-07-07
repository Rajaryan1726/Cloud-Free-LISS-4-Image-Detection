
"""
Dataset class for LISS-IV synthetic (cloudy, clean) pairs.
Reuses the same normalization convention as SEN12MS-CR so the pretrained
model transfers directly.
"""
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from . import config as cfg


def _read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)   # (3, H, W)


def _normalize_liss4(arr: np.ndarray) -> np.ndarray:
    """
    LISS-IV raw DN values are ~10-bit (0-1023 typically), unlike SEN12MS-CR's
    reflectance*10000 scale. Normalize using the actual LISS-IV data range
    so values land in [0,1] consistent with what the model expects.
    """
    arr = np.clip(arr, 0, cfg.LISS4_CLIP_MAX)
    return arr / cfg.LISS4_CLIP_MAX


class LISS4SyntheticDataset(Dataset):
    def __init__(self, manifest_rows):
        self.rows = manifest_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        cloudy = _normalize_liss4(_read_tif(row["cloudy_path"]))
        clean = _normalize_liss4(_read_tif(row["clean_path"]))

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clean": torch.from_numpy(clean),
            "patch_id": row["patch_id"],
        }