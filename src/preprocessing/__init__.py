"""
Preprocessing package.

This package contains all preprocessing utilities used throughout the
Cloud-Free Image Reconstruction project.

Currently exposed:
    - SEN12MSCRDataset : Main PyTorch dataset for loading SEN12MS-CR samples.
"""

from .dataset import SEN12MSCRDataset

__all__ = [
    "SEN12MSCRDataset",
]