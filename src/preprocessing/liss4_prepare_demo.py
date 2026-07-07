
"""
Lightweight variant: only regenerates patches for partial_cloudy and cloudy
categories (skips cloud_free, which we don't need for the real-cloudy demo).
"""
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from . import config as cfg

DEMO_CATEGORIES = ["partial_cloudy", "cloudy"]


def _find_scene_folders(category_dir: Path):
    if not category_dir.exists():
        return []
    return [p.parent for p in category_dir.rglob("BAND2.tif")]


def generate_patches(scene_folder: Path, scene_id: str, category: str,
                      patch_size: int, out_patch_dir: Path):
    band_paths = [scene_folder / f"{b}.tif" for b in cfg.LISS4_BANDS]
    for bp in band_paths:
        if not bp.exists():
            raise FileNotFoundError(f"Missing {bp}")

    srcs = [rasterio.open(bp) for bp in band_paths]
    try:
        height, width = srcs[0].height, srcs[0].width
        dtype = srcs[0].dtypes[0]

        n_rows = height // patch_size
        n_cols = width // patch_size
        out_patch_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in range(n_rows):
            for c in range(n_cols):
                window = Window(c * patch_size, r * patch_size, patch_size, patch_size)
                patch_data = np.stack(
                    [src.read(1, window=window) for src in srcs], axis=0
                )
                if np.mean(patch_data == 0) > 0.5:
                    continue

                patch_profile = srcs[0].profile.copy()
                patch_profile.update(
                    count=3, dtype=dtype,
                    height=patch_size, width=patch_size,
                    transform=rasterio.windows.transform(window, srcs[0].transform),
                    compress="lzw",
                )

                patch_id = f"{scene_id}_r{r}_c{c}"
                patch_path = out_patch_dir / f"{patch_id}.tif"
                with rasterio.open(patch_path, "w", **patch_profile) as dst:
                    dst.write(patch_data)

                rows.append({
                    "patch_id": patch_id,
                    "scene_id": scene_id,
                    "category": category,
                    "patch_path": str(patch_path),
                })
        return rows
    finally:
        for src in srcs:
            src.close()


def main():
    all_rows = []
    demo_out_dir = cfg.DATA_ROOT / "processed" / "liss4_demo_patches"

    for category in DEMO_CATEGORIES:
        category_dir = cfg.LISS4_CATEGORIES[category]
        scene_folders = _find_scene_folders(category_dir)
        print(f"[{category}] found {len(scene_folders)} scene(s)")

        for scene_folder in scene_folders:
            scene_id = scene_folder.name
            patch_out_dir = demo_out_dir / category
            print(f"  Processing {scene_id} ...")
            try:
                rows = generate_patches(scene_folder, scene_id, category, cfg.LISS4_PATCH_SIZE, patch_out_dir)
            except FileNotFoundError as e:
                print(f"    !! SKIPPED: {e}")
                continue
            print(f"    -> generated {len(rows)} patches")
            all_rows.extend(rows)

    manifest_path = cfg.LISS4_MANIFEST_DIR / "liss4_demo_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id", "scene_id", "category", "patch_path"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest saved -> {manifest_path}")
    print(f"Total demo patches: {len(all_rows)}")


if __name__ == "__main__":
    main()