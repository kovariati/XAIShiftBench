"""Dataset adapters used by XAIShiftBench."""

from .heart_disease import (
    HEART_COLUMNS,
    SITE_FILE_MAP,
    HeartDataset,
    load_heart_sites,
    validate_heart_dataset,
)

__all__ = [
    "HEART_COLUMNS",
    "SITE_FILE_MAP",
    "HeartDataset",
    "load_heart_sites",
    "validate_heart_dataset",
]
