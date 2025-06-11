import os
import random
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from Utils.load_and_resize import load_and_resize
from Utils.convolution import upstream_only_KuT
import matplotlib.pyplot as plt


def generate_post(
    scene_ids: list[str],
    folders: dict[str, str],
    KERNEL: np.ndarray,
    Bbase: float,
    Beta1: float,
    Beta2: float,
    coarse_shape: tuple[int,int] = (8, 8),
    noise_type: str = "gaussian",    # "gaussian" or "none"
    noise_sd: float = 0.0,           # for Gaussian noise
    results_dir: str | Path | None = None
) -> None:
    """
    Simplified coarse‐resolution post‐outcome generator using only load_and_resize:
      - DEM, Capital, Treatment, Outcome all resized directly to `coarse_shape`.
      - Compute θ(x), KuT_up, ITE, then post = outcome + ITE (+ noise).
      - Save coarse TIFFs and comparison PDFs.
    """
    # Setup directories
    if results_dir is None:
        base = (
            '~/Library/CloudStorage/Box-Box/Caltech Research/Scripts/'
            'ImageTRT/Synthetic Data Generation/Results'
        )
        results_dir = Path(os.path.expanduser(base))
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    treatment_dir = results_dir / 'Treatment'
    outcome_dir   = results_dir / 'Outcome'
    ite_dir       = results_dir / 'ITE'
    ite_dir.mkdir(parents=True, exist_ok=True)
    post_dir      = results_dir / 'Outcome_Post'
    post_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    pdf_ids = set(random.sample(scene_ids, min(5, len(scene_ids))))

    for sid in scene_ids:
        # Paths
        paths = {
            'dem':     Path(folders['dem'])      / f"DEM_{sid}.tiff",
            'cap':     Path(folders['cap'])      / f"CAPITAL_1996_{sid}.tiff",
            'treat':   treatment_dir            / f"scene_{sid}_synthetic_gaussian.tiff",
            'actual':  outcome_dir              / f"scene_{sid}_synthetic_gaussian.tiff"
        }

        # Load & resize
        dem_r    = load_and_resize(paths['dem'],    coarse_shape, Image.BILINEAR)
        cap_r    = load_and_resize(paths['cap'],    coarse_shape, Image.BILINEAR)
        treat_r  = load_and_resize(paths['treat'],  coarse_shape, Image.NEAREST)
        actual_r = load_and_resize(paths['actual'], coarse_shape, Image.NEAREST)

        if any(x is None for x in (dem_r, cap_r, treat_r, actual_r)):
            logging.warning(f"Skipping {sid}: missing inputs for coarse gen.")
            continue

        # Cast
        dem    = dem_r.astype(np.float32)
        cap    = cap_r.astype(np.float32)
        treat = treat_r.astype(np.float32)
        actual = actual_r.astype(np.float32)

        # Compute θ(x)
        dem_safe = np.maximum(dem, 1.0)
        theta = Bbase * (1.0
                         + Beta1 * np.log1p(cap)
                         + Beta2 / dem_safe)

        # Compute upstream spillover on coarse grid
        KuT_up = upstream_only_KuT(treat, dem, KERNEL)

        # ITE & post outcome
        ITE  = theta * KuT_up
        post = actual + ITE
        if noise_type == "gaussian":
            post += np.random.normal(0, noise_sd, size=post.shape)
        elif noise_type != "none":
            raise ValueError("noise_type must be 'gaussian' or 'none'")

        # Save TIFF
        out_tif = post_dir / f"scene_{sid}_post_{noise_type}.tiff"
        Image.fromarray(post.astype(np.float32), mode='F').save(out_tif)
        logging.info(f"[{sid}] Saved post outcome coarse → {out_tif.name}")

        # Save ITE
        ite_tif = ite_dir / f"scene_{sid}_ITE.tiff"
        Image.fromarray(ITE.astype(np.float32), mode='F').save(ite_tif)
        logging.info(f"[{sid}] Saved ITE coarse → {ite_tif.name}")

        # PDF compare
        if sid in pdf_ids:
            # 3-panel: Baseline, ITE, Post
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Baseline
            im0 = axes[0].imshow(actual, cmap='viridis')
            axes[0].set_title('Baseline')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # ITE
            im1 = axes[1].imshow(ITE, cmap='viridis')
            axes[1].set_title('ITE')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # Post
            im2 = axes[2].imshow(post, cmap='viridis')
            axes[2].set_title('Post')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            fig.suptitle(f"{sid}: Baseline, ITE & Post (coarse {coarse_shape})", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            pdf_path = results_dir / f"{sid}_pre_post_comparison_{noise_type}.pdf"
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"[{sid}] Saved comparison PDF → {pdf_path.name}")

