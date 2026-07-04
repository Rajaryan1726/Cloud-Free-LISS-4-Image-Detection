"""
SEN12MS-CR dataset loader + manifest builder.

Works even when s2 (clean) isn't downloaded yet -- manifest will just mark
those rows as incomplete, and the Dataset class raises a clear error only
when you actually try to load a missing modality.

Filename convention in this dataset differs per modality
(e.g. ROIs1158_spring_s1_1_p100.tif vs ROIs1158_spring_s2_cloudy_1_p100.tif),
so matching is done on the (roi, patch) key extracted via regex, not on the
raw filename.
"""
import csv
import re
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from . import config as cfg

# matches "..._<roi>_p<patch>.tif" regardless of modality tag in the filename
_KEY_PATTERN = re.compile(r"_(\d+)_p(\d+)\.tif$")


def _index_by_key(folder) -> dict:
    """
    Scan a modality folder and return {(roi, patch): filename}.
    Ignores the modality tag (s1 / s2_cloudy / s2) in the filename so that
    files across folders can be matched by (roi, patch) instead of by
    exact filename.
    """
    index = {}
    if not folder.exists():
        return index
    for f in folder.rglob("*.tif"):   # recursive: files live inside per-ROI subfolders (e.g. s1_1/, s1_100/)
        m = _KEY_PATTERN.search(f.name)
        if m:
            roi, patch = m.groups()
            index[(roi, patch)] = f   # store full Path, not just name, since files are nested
    return index


def build_manifest(season: str, save_csv: bool = True):
    """
    Scan raw/{season}_s1, raw/{season}_s2_cloudy, raw/{season}_s2 (via
    cfg.modality_dir), match patches by (roi, patch) key, and write a
    manifest CSV. Works right now with only s1 + s2_cloudy present;
    's2' (clean) column will just read empty/False until that tarball is
    extracted -- has_clean flips to True automatically once it is.
    """
    s1_dir        = cfg.modality_dir(season, "s1")
    s2_cloudy_dir = cfg.modality_dir(season, "s2_cloudy")
    s2_dir        = cfg.modality_dir(season, "s2")

    s1_idx        = _index_by_key(s1_dir)
    s2_cloudy_idx = _index_by_key(s2_cloudy_dir)
    s2_idx        = _index_by_key(s2_dir)

    # only s1+s2_cloudy required to be usable right now; s2 optional for now
    common_keys = s1_idx.keys() & s2_cloudy_idx.keys()
    print(f"[{season}] s1={len(s1_idx)}  s2_cloudy={len(s2_cloudy_idx)}  "
          f"s2(clean)={len(s2_idx)}  matched(s1+s2_cloudy)={len(common_keys)}")

    if not common_keys:
        print(f"[{season}] WARNING: no matches found. Checked:\n"
              f"  s1_dir        = {s1_dir}\n"
              f"  s2_cloudy_dir = {s2_cloudy_dir}\n"
              f"Make sure these folders exist and contain '..._<roi>_p<patch>.tif' files.")
        return []

    rows = []
    for key in sorted(common_keys, key=lambda k: (int(k[0]), int(k[1]))):
        roi, patch = key
        rows.append({
            "patch_id": f"{roi}_p{patch}",
            "season": season,
            "s1_path": str(s1_idx[key]),
            "s2_cloudy_path": str(s2_cloudy_idx[key]),
            "s2_path": str(s2_idx[key]) if key in s2_idx else "",
            "has_clean": key in s2_idx,
        })

    if save_csv:
        cfg.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        out_path = cfg.MANIFEST_DIR / f"{season}_manifest.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Manifest saved -> {out_path}")

    return rows


def _read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)   # (C, H, W)


def _normalize_s1(arr):
    arr = np.clip(arr, cfg.S1_CLIP_MIN, cfg.S1_CLIP_MAX)
    return (arr - cfg.S1_CLIP_MIN) / (cfg.S1_CLIP_MAX - cfg.S1_CLIP_MIN)  # -> [0,1]


def _normalize_s2(arr):
    arr = np.clip(arr, cfg.S2_CLIP_MIN, cfg.S2_CLIP_MAX)
    return arr / cfg.S2_CLIP_MAX  # -> [0,1]


class SEN12MSCRDataset(Dataset):
    """
    require_clean=False -> usable right now with only s1 + s2_cloudy
                            (e.g. for building a masking/statistics pass).
    require_clean=True  -> will only yield rows where the clean s2 patch
                            exists (use this once spring s2.tar.gz is extracted).
    """
    def __init__(self, manifest_rows, require_clean: bool = False):
        self.rows = [r for r in manifest_rows if (not require_clean or r["has_clean"])]
        self.require_clean = require_clean

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        s1 = _normalize_s1(_read_tif(row["s1_path"]))
        s2_cloudy = _normalize_s2(_read_tif(row["s2_cloudy_path"]))

        sample = {
            "s1": torch.from_numpy(s1),
            "s2_cloudy": torch.from_numpy(s2_cloudy),
            "patch_id": row["patch_id"],
        }

        if row["has_clean"]:
            s2_clean = _normalize_s2(_read_tif(row["s2_path"]))
            sample["s2_clean"] = torch.from_numpy(s2_clean)
        elif self.require_clean:
            raise FileNotFoundError(f"clean s2 missing for {row['patch_id']}")

        return sample


if __name__ == "__main__":
    # sanity check with what's downloaded right now (s1 + s2_cloudy, spring)
    rows = build_manifest("spring")
    if rows:
        ds = SEN12MSCRDataset(rows, require_clean=False)
        print(f"Dataset size (require_clean=False): {len(ds)}")
        sample = ds[0]
        print({k: (v.shape if torch.is_tensor(v) else v) for k, v in sample.items()})