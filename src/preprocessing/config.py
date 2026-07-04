"""
Central configuration for SEN12MS-CR preprocessing + LISS-IV fine-tuning pipeline.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # repo root
DATA_ROOT    = PROJECT_ROOT / "data"

RAW_DIR       = DATA_ROOT / "raw"          # ROIs1158_spring_s1/ etc live directly here
PATCH_DIR     = DATA_ROOT / "patches"      # optional: cropped/tiled patches go here
PROCESSED_DIR = DATA_ROOT / "processed"    # normalized .npy / .pt tensors go here
SPLIT_DIR     = DATA_ROOT / "splits"       # train/val/test csv or txt lists
MANIFEST_DIR  = DATA_ROOT / "manifests"
LISS4_DIR     = DATA_ROOT / "raw" / "LISS-IV"   # bhoonidhi downloads (baad me)

# season name -> ROI prefix used in TUM filenames
SEASON_PREFIX = {
    "spring": "ROIs1158_spring",
    "summer": "ROIs1868_summer",
    "fall":   "ROIs1970_fall",
    "winter": "ROIs2017_winter",
}

# modality suffix as it appears after the season prefix in folder names
MODALITY_SUFFIX = {
    "s1":        "s1",
    "s2_cloudy": "s2_cloudy",
    "s2":        "s2",   # clean/label -- may not exist yet, handled gracefully
}


def modality_dir(season: str, modality: str) -> Path:
    """e.g. modality_dir('spring', 's2_cloudy') -> data/raw/ROIs1158_spring_s2_cloudy"""
    return RAW_DIR / f"{SEASON_PREFIX[season]}_{MODALITY_SUFFIX[modality]}"


# ---------------------------------------------------------------------------
# SENSOR SPECS
# ---------------------------------------------------------------------------
S1_BANDS   = 2      # VV, VH (sigma0, dB)
S2_BANDS   = 13     # full Sentinel-2 stack as shipped in SEN12MS-CR
PATCH_SIZE = 256    # native patch size of SEN12MS-CR tiles

# clip ranges used for normalization (standard for SEN12MS-CR, per DSen2-CR paper)
S1_CLIP_MIN, S1_CLIP_MAX = -25.0, 0.0      # dB
S2_CLIP_MIN, S2_CLIP_MAX = 0.0, 10000.0    # reflectance * 10000

# ---------------------------------------------------------------------------
# LISS-IV <-> Sentinel-2 band mapping (fine-tuning stage)
# LISS-IV: Green, Red, NIR only (no blue/SWIR)
# S2 13-band order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
# ---------------------------------------------------------------------------
S2_BAND_INDEX_FOR_LISS4 = {"B3": 2, "B4": 3, "B8": 7}   # Green, Red, NIR

# ---------------------------------------------------------------------------
# SPLIT
# ---------------------------------------------------------------------------
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# HARDWARE-AWARE TRAINING DEFAULTS
# ---------------------------------------------------------------------------
LOCAL_PATCH_SIZE, LOCAL_BATCH_SIZE = 128, 4     # RTX 3050 (4GB)
CLOUD_PATCH_SIZE, CLOUD_BATCH_SIZE = 256, 16    # Kaggle/Colab T4