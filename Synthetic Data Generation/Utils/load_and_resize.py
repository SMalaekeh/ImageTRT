import logging
from pathlib import Path
import numpy as np
import rasterio as rio
from PIL import Image

def load_and_resize(path: Path, shape: tuple[int, int], resample) -> np.ndarray:
    """
    Load a single-band image from `path`, resize to `shape`, and return as a numpy array.
    """
    if not path.exists():
        logging.warning(f"File not found: {path}")
        return None
    with rio.open(path) as src:
        arr = src.read(1)
    dtype = np.uint8 if resample == Image.NEAREST else np.float32
    img = Image.fromarray(arr)
    resized = img.resize(shape, resample)
    return np.array(resized, dtype=dtype)
