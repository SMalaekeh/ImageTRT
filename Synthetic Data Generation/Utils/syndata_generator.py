import os
import random
import logging
from pathlib import Path
import numpy as np
import rasterio as rio
from PIL import Image
import matplotlib.pyplot as plt
import load_and_resize

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function element-wise."""
    return 1 / (1 + np.exp(-x))


def generate_synthetic_wetland(
    scene_ids: list[str],
    folders: dict[str, str],
    logit_pipe,
    threshold: float = 0.5,
    target_shape: tuple[int, int] = (256, 256),
    results_dir: str | Path | None = None
) -> None:
    """
    Generate and save synthetic wetland TIFFs and comparison PDFs using a logistic pipeline.
    """
    # Setup output directories
    results_dir = Path(results_dir) if results_dir else Path(
        os.path.expanduser(
            '~/Library/CloudStorage/Box-Box/Caltech Research/Scripts/'
            'ImageTRT/Synthetic Data Generation/Results'
        )
    )
    treatment_dir = results_dir / 'Treatment'
    treatment_dir.mkdir(parents=True, exist_ok=True)

    # Extract model params
    lr = logit_pipe.named_steps['logreg']
    b0 = lr.intercept_[0]
    b1, b2 = lr.coef_[0]

    # Select scenes for PDF comparison
    pdf_ids = random.sample(scene_ids, min(5, len(scene_ids)))

    for sid in scene_ids:
        # Load and resize inputs
        wet = load_and_resize(Path(folders['wet']) / f"WETLAND_DEV_1996_2016_{sid}.tiff", target_shape, Image.NEAREST)
        dem = load_and_resize(Path(folders['dem']) / f"DEM_{sid}.tiff", target_shape, Image.BILINEAR)
        cap = load_and_resize(Path(folders['cap']) / f"CAPITAL_1996_{sid}.tiff", target_shape, Image.BILINEAR)
        if None in (wet, dem, cap):
            continue

        # Compute predictions
        lin = b0 + b1 * dem + b2 * cap
        prob = _sigmoid(lin)
        synth = (prob >= threshold).astype(np.uint8)

        # Save synthetic map as TIFF
        out_tif = treatment_dir / f"scene_{sid}_synthetic.tiff"
        Image.fromarray((synth * 255).astype(np.uint8)).save(out_tif)

        # Save comparison PDF for selected scenes
        if sid in pdf_ids:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(wet, cmap='gray'); ax1.axis('off')
            ax1.set_title(f"Actual {sid}")
            ax2.imshow(synth, cmap='gray'); ax2.axis('off')
            ax2.set_title(f"Synthetic {sid}")
            fig.suptitle(f"Scene {sid}: Actual vs Synthetic", fontsize=16)
            plt.tight_layout(rect=[0,0,1,0.93])
            pdf_out = results_dir / f"scene_{sid}_synthetic_comparison.pdf"
            fig.savefig(pdf_out, bbox_inches='tight')
            plt.close(fig)

    logging.info(f"Synthetic wetland TIFFs and PDFs saved under {results_dir}")
