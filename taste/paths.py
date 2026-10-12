"""Root directories of the release. Relative paths in configs and CLI arguments resolve against these."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TASTE_DATA_ROOT", "data"))
CKPT_ROOT = Path(os.environ.get("TASTE_CKPT_ROOT", "checkpoints"))
# External repositories used by annotation.
CLAMP3_ROOT = Path(os.environ.get("TASTE_CLAMP3_ROOT", "external/clamp3"))
ARIA_ROOT = Path(os.environ.get("TASTE_ARIA_ROOT", "external/aria"))


def data(*parts) -> Path:
    """Path under TASTE_DATA_ROOT; an absolute argument is returned unchanged."""
    return DATA_ROOT.joinpath(*parts)


def ckpt(*parts) -> Path:
    """Path under TASTE_CKPT_ROOT; an absolute argument is returned unchanged."""
    return CKPT_ROOT.joinpath(*parts)
