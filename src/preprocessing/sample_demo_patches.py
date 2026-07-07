"""
Randomly samples N patches per category from the demo manifest
and copies them into a small folder for Kaggle upload.
"""
import csv
import random
import shutil
from pathlib import Path

from . import config as cfg

N_PER_CATEGORY = 20
SEED = 42

def main():
    manifest_path = cfg.LISS4_MANIFEST_DIR / "liss4_demo_manifest.csv"
    out_dir = cfg.DATA_ROOT / "processed" / "liss4_demo_sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_category = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_category.setdefault(row["category"], []).append(row)

    random.seed(SEED)
    sampled_rows = []

    for category, rows in rows_by_category.items():
        k = min(N_PER_CATEGORY, len(rows))
        chosen = random.sample(rows, k)
        cat_out_dir = out_dir / category
        cat_out_dir.mkdir(parents=True, exist_ok=True)

        for row in chosen:
            src_path = Path(row["patch_path"])
            dst_path = cat_out_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            row["patch_path"] = str(dst_path)
            sampled_rows.append(row)

        print(f"[{category}] sampled {k} patches -> {cat_out_dir}")

    sample_manifest_path = cfg.LISS4_MANIFEST_DIR / "liss4_demo_sample_manifest.csv"
    with open(sample_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patch_id", "scene_id", "category", "patch_path"])
        writer.writeheader()
        writer.writerows(sampled_rows)

    print(f"\nSample manifest saved -> {sample_manifest_path}")
    print(f"Total sampled patches: {len(sampled_rows)}")


if __name__ == "__main__":
    main()