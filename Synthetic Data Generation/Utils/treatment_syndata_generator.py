import os
import random
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from Utils.load_and_resize import load_and_resize
import matplotlib.pyplot as plt

def generate_synthetic_treatment(
    scene_ids: list[str],
    folders: dict[str, str],
    logit_pipe,
    threshold: float = 0.5,
    target_shape: tuple[int, int] = (256, 256),
    noise_type: str = "gaussian",
    noise_sd: float = 0.2,
    n_trials: int = 1,
    results_dir: str | Path | None = None
) -> None:
    """
    Generate synthetic wetland maps using a logistic regression model trained with
    DEM, Capital, and their scene-level means as features.
    """
    # Setup output directory
    results_dir = Path(results_dir) if results_dir else Path(
        os.path.expanduser(
            '~/Library/CloudStorage/Box-Box/Hetwet_Data/Synthetic'
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    treatment_dir = results_dir / 'Treatment'
    treatment_dir.mkdir(parents=True, exist_ok=True)

    # Sample a few scenes for plotting
    random.seed(42)
    pdf_ids = random.sample(scene_ids, min(5, len(scene_ids)))

    for sid in scene_ids:
        # Load inputs
        wet = load_and_resize(Path(folders['wet']) / f"WETLAND_DEV_1996_2016_{sid}.tiff",
                              target_shape, Image.NEAREST)
        dem = load_and_resize(Path(folders['dem']) / f"DEM_{sid}.tiff",
                              target_shape, Image.BILINEAR)
        cap = load_and_resize(Path(folders['cap']) / f"CAPITAL_1996_{sid}.tiff",
                              target_shape, Image.BILINEAR)
        if wet is None or dem is None or cap is None:
            logging.warning(f"Skipping {sid}: failed to load inputs")
            continue

        # Flatten pixel-level features
        dem_flat = dem.ravel()
        cap_flat = cap.ravel()

        # Scene-level means
        # dem_mean = np.mean(dem_flat)
        # cap_mean = np.mean(cap_flat)

        # Repeat means to match pixel count
        # dem_mean_col = np.full_like(dem_flat, dem_mean)
        # cap_mean_col = np.full_like(cap_flat, cap_mean)

        # Final input features: [DEM, DEM_mean, Capital, Capital_mean]
        # X_scene = np.column_stack([dem_flat, dem_mean_col, cap_flat, cap_mean_col])
        X_scene = np.column_stack([dem_flat, cap_flat])
        probs = logit_pipe.predict_proba(X_scene)[:, 1].reshape(target_shape)

        # Apply noise mode
        if noise_type == "deterministic":
            synth = (probs >= threshold).astype(np.uint8)

        elif noise_type == "gaussian":
            probs_noisy = np.clip(
                probs + np.random.normal(0, noise_sd, size=probs.shape),
                0, 1
            )
            synth = (probs_noisy >= threshold).astype(np.uint8)

        elif noise_type == "bernoulli":
            synth = np.random.binomial(n_trials, probs, size=probs.shape)
            if n_trials > 1:
                synth = (synth / n_trials >= threshold).astype(np.uint8)
            else:
                synth = synth.astype(np.uint8)

        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # Save synthetic TIFF
        out_tif = treatment_dir / f"treatment_scene_{sid}_{noise_type}.tiff"
        Image.fromarray(synth.astype(np.uint8)).save(out_tif)

        # Save visual comparison
        if sid in pdf_ids:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(wet, cmap='gray'); ax1.axis('off'); ax1.set_title(f"Actual {sid}")
            ax2.imshow(synth, cmap='gray'); ax2.axis('off'); ax2.set_title(f"Synthetic {sid}")
            fig.suptitle(f"{sid} — {noise_type}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf_out = results_dir / f"{sid}_treatment_comparison_{noise_type}.pdf"
            fig.savefig(pdf_out, bbox_inches='tight')
            plt.close(fig)

        logging.info(f"[{noise_type}] Saved synthetic for {sid}")

    logging.info(f"All outputs written to {results_dir}")