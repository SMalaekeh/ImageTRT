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
    noise_type: str = "gaussian",    # "deterministic", "gaussian", or "bernoulli"
    noise_sd: float = 0.2,           # used only for gaussian
    n_trials: int = 1,               # used only for bernoulli
    results_dir: str | Path | None = None
) -> None:
    """
    Generate synthetic wetland maps under three modes:
      - deterministic:  Hard threshold on model's P(y=1)
      - gaussian:       Add Gaussian noise to P(y=1) before threshold
      - bernoulli:      Draw Bernoulli( p_i ) for each pixel

    Uses logit_pipe.predict_proba(...) internally (which applies sigmoid).
    """
    # -- setup directories --
    results_dir = Path(results_dir) if results_dir else Path(
        os.path.expanduser(
            '~/Library/CloudStorage/Box-Box/Caltech Research/Scripts/'
            'ImageTRT/Synthetic Data Generation/Results'
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    treatment_dir = results_dir / 'Treatment'
    treatment_dir.mkdir(parents=True, exist_ok=True)

    # fix seed for reproducibility
    random.seed(42)
    pdf_ids = random.sample(scene_ids, min(5, len(scene_ids)))

    for sid in scene_ids:
        # load & resize rasters
        wet = load_and_resize(Path(folders['wet']) / f"WETLAND_DEV_1996_2016_{sid}.tiff",
                               target_shape, Image.NEAREST)
        dem = load_and_resize(Path(folders['dem']) / f"DEM_{sid}.tiff",
                               target_shape, Image.BILINEAR)
        cap = load_and_resize(Path(folders['cap']) / f"CAPITAL_1996_{sid}.tiff",
                               target_shape, Image.BILINEAR)
        if wet is None or dem is None or cap is None:
            logging.warning(f"Skipping {sid}: failed to load inputs")
            continue

        # flatten and get model probabilities
        X_scene = np.column_stack([dem.ravel(), cap.ravel()])
        probs   = logit_pipe.predict_proba(X_scene)[:, 1].reshape(target_shape)

        # choose noise mode
        if noise_type == "deterministic":
            # hard threshold
            synth = (probs >= threshold).astype(np.uint8)

        elif noise_type == "gaussian":
            # jitter the probs, then threshold
            probs_noisy = np.clip(
                probs + np.random.normal(0, noise_sd, size=probs.shape),
                0, 1
            )
            synth = (probs_noisy >= threshold).astype(np.uint8)

        elif noise_type == "bernoulli":
            # Bernoulli draw per pixel
            synth = np.random.binomial(n_trials, probs, size=probs.shape)
            # if n_trials>1 you'd get 0…n_trials; for binary, n_trials=1
            if n_trials > 1:
                synth = (synth / n_trials >= threshold).astype(np.uint8)
            else:
                synth = synth.astype(np.uint8)

        else:
            raise ValueError(f"Unknown noise_type: {noise_type!r}")

        # save synthetic TIFF
        out_tif = treatment_dir / f"scene_{sid}_synthetic_{noise_type}.tiff"
        Image.fromarray((synth).astype(np.uint8)).save(out_tif)

        # save a PDF comparison for a few scenes
        if sid in pdf_ids:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(wet, cmap='gray'); ax1.axis('off'); ax1.set_title(f"Actual {sid}")
            ax2.imshow(synth, cmap='gray'); ax2.axis('off'); ax2.set_title(f"Synthetic {sid}")
            fig.suptitle(f"{sid} — {noise_type}", fontsize=16)
            plt.tight_layout(rect=[0,0,1,0.93])
            pdf_out = results_dir / f"{sid}_treatment_comparison_{noise_type}.pdf"
            fig.savefig(pdf_out, bbox_inches='tight')
            plt.close(fig)

        logging.info(f"[{noise_type}] Saved synthetic for {sid}")

    logging.info(f"All outputs written to {results_dir}")
