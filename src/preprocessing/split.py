"""
ROI-level train/val/test split for the SEN12MS-CR manifest.

Splitting is done at the ROI level (not patch level) so that all patches
belonging to the same ROI end up in the same split -- this avoids data
leakage (model seeing patches from the same geographic area in both
train and val/test).

Reads:  data/manifests/{season}_manifest.csv   (built by dataset.py)
Writes: data/splits/{season}_train.csv
        data/splits/{season}_val.csv
        data/splits/{season}_test.csv
"""
import csv
import random
from collections import defaultdict

from . import config as cfg


def _load_manifest(season: str) -> list[dict]:
    manifest_path = cfg.MANIFEST_DIR / f"{season}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run "
            f"'python -m src.preprocessing.dataset' first to build it."
        )
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def _roi_of(patch_id: str) -> str:
    # patch_id format: "{roi}_p{patch}" e.g. "1_p30" -> roi "1"
    return patch_id.split("_p")[0]


def split_manifest(season: str, save_csv: bool = True):
    rows = _load_manifest(season)

    # group patch rows by ROI
    rois_to_rows = defaultdict(list)
    for row in rows:
        rois_to_rows[_roi_of(row["patch_id"])].append(row)

    rois = sorted(rois_to_rows.keys(), key=int)
    rng = random.Random(cfg.SPLIT_SEED)
    rng.shuffle(rois)

    n = len(rois)
    n_train = int(n * cfg.TRAIN_RATIO)
    n_val   = int(n * cfg.VAL_RATIO)
    # remainder goes to test, so ratios always sum correctly even with rounding
    train_rois = set(rois[:n_train])
    val_rois   = set(rois[n_train:n_train + n_val])
    test_rois  = set(rois[n_train + n_val:])

    def _collect(roi_set):
        out = []
        for roi in roi_set:
            out.extend(rois_to_rows[roi])
        return out

    splits = {
        "train": _collect(train_rois),
        "val":   _collect(val_rois),
        "test":  _collect(test_rois),
    }

    print(f"[{season}] total ROIs={n}  ->  "
          f"train_rois={len(train_rois)} ({len(splits['train'])} patches)  "
          f"val_rois={len(val_rois)} ({len(splits['val'])} patches)  "
          f"test_rois={len(test_rois)} ({len(splits['test'])} patches)")

    if save_csv:
        cfg.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = rows[0].keys()
        for split_name, split_rows in splits.items():
            out_path = cfg.SPLIT_DIR / f"{season}_{split_name}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(split_rows)
            print(f"Saved -> {out_path}")

    return splits


if __name__ == "__main__":
    split_manifest("spring")