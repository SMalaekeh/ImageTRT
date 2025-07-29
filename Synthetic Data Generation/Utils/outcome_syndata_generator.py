import os
import random
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from Utils.load_and_resize import load_and_resize
import matplotlib.pyplot as plt

def generate_synthetic_outcome(
    scene_ids: list[str],
    folders: dict[str, str],
    reg_pipe,
    target_shape: tuple[int, int] = (256, 256),
    noise_type: str = "gaussian",   # "gaussian" or "none"
    noise_sd: float = 0.2,          # only for gaussian
    results_dir: str | Path | None = None,
    verbose: bool = True
) -> None:
    """
    Using a fitted regression pipeline (`reg_pipe`), generate and save:
      - Actual ΔLogClaims TIFFs
      - Synthetic ΔLogClaims TIFFs (with optional Gaussian noise)
      - PDFs comparing actual vs synthetic for 5 random scenes
    """
    # --- setup directories ---
    results_dir = Path(results_dir) if results_dir else Path(
        os.path.expanduser(
            '~/Library/CloudStorage/Box-Box/Hetwet_Data/Synthetic'
        )
    )
    outcome_dir = results_dir / 'Outcome'
    outcome_dir.mkdir(parents=True, exist_ok=True)

    # reproducible selection of 5 scenes for PDF
    random.seed(42)
    pdf_ids = set(random.sample(scene_ids, min(5, len(scene_ids))))

    for sid in scene_ids:
        # load & resize
        dem = load_and_resize(Path(folders['dem']) / f"DEM_{sid}.tiff",
                              target_shape, Image.BILINEAR)
        cap = load_and_resize(Path(folders['cap']) / f"CAPITAL_1996_{sid}.tiff",
                              target_shape, Image.BILINEAR)
        c96 = load_and_resize(Path(folders['claims_96']) / f"LOG_CLAIMS_1996_{sid}.tiff",
                              target_shape, Image.BILINEAR)

        # skip if any loading failed
        if any(x is None for x in (dem, cap, c96)):
            logging.warning(f"Skipping {sid}: failed to load inputs")
            continue

        # actual continuous outcome
        actual = (c96).astype(np.float32)

        # predict via pipeline
        X_scene = np.column_stack([dem.ravel(), cap.ravel()])
        pred_flat = reg_pipe.predict(X_scene)

        # optional noise
        if noise_type == "gaussian":
            pred_flat = pred_flat + np.random.normal(
                loc=0, scale=noise_sd, size=pred_flat.shape
            )
        elif noise_type != "none":
            raise ValueError("noise_type must be 'gaussian' or 'none'")

        pred = pred_flat.reshape(target_shape).astype(np.float32)

        # save synthetic as 32-bit TIFF
        synth_tif  = outcome_dir / f"outcome_scene_{sid}.tiff"
        Image.fromarray(pred,   mode='F').save(synth_tif)

        if verbose:
            logging.info(f"[{sid}] Saved synthetic → {synth_tif.name}")

        # for a handful of scenes, also write a PDF compare
        if sid in pdf_ids:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im1 = ax1.imshow(actual, cmap='viridis')
            ax1.set_title(f"Actual Baseline Log Claims {sid}")
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            im2 = ax2.imshow(pred, cmap='viridis')
            ax2.set_title(f"Synthetic Baseline Log Claims {sid}")
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            fig.suptitle(f"{sid}: Actual vs Synthetic ({noise_type})", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            pdf_path = results_dir / f"{sid}_outcome_comparison.pdf"
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"[{sid}] Saved comparison PDF → {pdf_path.name}")

    logging.info(f"All outcome TIFFs and PDFs written to {outcome_dir}")
