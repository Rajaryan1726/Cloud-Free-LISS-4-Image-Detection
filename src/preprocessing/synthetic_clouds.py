
"""
Synthetic cloud simulation for LISS-IV fine-tuning.

Uses Gaussian-blurred random noise (instead of Perlin noise, to avoid
needing a C++ compiler on Windows) to generate realistic cloud-blob masks.
Produces paired (synthetic_cloudy, clean) training data.

Usage:
    python -m src.preprocessing.synthetic_clouds
    python -m src.preprocessing.synthetic_clouds --limit 3000
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

from . import config as cfg

SYNTH_OUT_DIR = cfg.DATA_ROOT / "processed" / "liss4_synthetic"
SYNTH_MANIFEST_PATH = cfg.LISS4_MANIFEST_DIR / "liss4_synthetic_manifest.csv"


def generate_cloud_mask(height: int, width: int, seed: int = None) -> np.ndarray:
    """
    Generate a smooth, blob-like cloud mask in [0,1] using Gaussian-blurred
    random noise -- visually similar to Perlin noise, no compiled deps needed.
    """
    rng = np.random.RandomState(seed)

    # random blob size per patch (bigger sigma = bigger, softer cloud blobs)
    sigma = rng.uniform(15, 40)

    raw_noise = rng.rand(height, width).astype(np.float32)
    mask = gaussian_filter(raw_noise, sigma=sigma)

    # normalize to [0,1]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # threshold to get a target coverage fraction (varies per patch)
    coverage_target = rng.uniform(0.15, 0.55)
    threshold = np.percentile(mask, (1 - coverage_target) * 100)
    mask = np.clip((mask - threshold) / (1 - threshold + 1e-8), 0, 1)

    # soften edges a bit more for realism
    mask = gaussian_filter(mask, sigma=3)

    return mask


def apply_synthetic_cloud(clean_patch: np.ndarray, cloud_mask: np.ndarray,
                           cloud_brightness: float = None) -> np.ndarray:
    """
    clean_patch: (3, H, W) native dtype (e.g. uint16)
    cloud_mask: (H, W) in [0,1], 1 = fully cloudy
    """
    if cloud_brightness is None:
        cloud_brightness = np.percentile(clean_patch, 98) * 1.15

    clean_f = clean_patch.astype(np.float32)
    cloud_layer = np.full_like(clean_f, cloud_brightness)

    mask_3ch = np.broadcast_to(cloud_mask, clean_f.shape)
    cloudy = clean_f * (1 - mask_3ch) + cloud_layer * mask_3ch

    if clean_patch.dtype == np.uint16:
        cloudy = np.clip(cloudy, 0, 65535)
    elif clean_patch.dtype == np.uint8:
        cloudy = np.clip(cloudy, 0, 255)

    return cloudy.astype(clean_patch.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of patches to process (default: all)")
    args = parser.parse_args()

    import pandas as pd

    manifest = pd.read_csv(cfg.LISS4_MANIFEST_DIR / "liss4_manifest.csv")
    clean_rows = manifest[manifest["category"] == "cloud_free"]

    if args.limit:
        clean_rows = clean_rows.sample(n=min(args.limit, len(clean_rows)), random_state=42)

    print(f"Processing {len(clean_rows)} cloud_free patches")

    SYNTH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    (SYNTH_OUT_DIR / "cloudy").mkdir(parents=True, exist_ok=True)
    (SYNTH_OUT_DIR / "clean").mkdir(parents=True, exist_ok=True)

    out_rows = []
    for idx, row in clean_rows.iterrows():
        clean_path = Path(row["patch_path"])
        if not clean_path.exists():
            continue

        with rasterio.open(clean_path) as src:
            clean_patch = src.read()
            profile = src.profile.copy()

        h, w = clean_patch.shape[1], clean_patch.shape[2]

        if np.mean(clean_patch == 0) > 0.3:
            continue

        cloud_mask = generate_cloud_mask(h, w, seed=idx)
        cloudy_patch = apply_synthetic_cloud(clean_patch, cloud_mask)

        patch_id = row["patch_id"]
        cloudy_path = SYNTH_OUT_DIR / "cloudy" / f"{patch_id}_synthcloudy.tif"
        clean_out_path = SYNTH_OUT_DIR / "clean" / f"{patch_id}_clean.tif"

        with rasterio.open(cloudy_path, "w", **profile) as dst:
            dst.write(cloudy_patch)
        with rasterio.open(clean_out_path, "w", **profile) as dst:
            dst.write(clean_patch)

        out_rows.append({
            "patch_id": patch_id,
            "cloudy_path": str(cloudy_path),
            "clean_path": str(clean_out_path),
        })

        if len(out_rows) % 500 == 0:
            print(f"  ... {len(out_rows)} synthetic pairs generated")

    with open(SYNTH_MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id", "cloudy_path", "clean_path"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. {len(out_rows)} synthetic (cloudy, clean) pairs saved.")
    print(f"Manifest -> {SYNTH_MANIFEST_PATH}")


if __name__ == "__main__":
    main()