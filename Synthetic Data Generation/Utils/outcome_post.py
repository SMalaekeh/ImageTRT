import os
import random
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from Utils.load_and_resize import load_and_resize
from Utils.convolution import upstream_only_KuT, downstream_only_KuT
import matplotlib.pyplot as plt

def to_coarse(arr256, target_shape=(8, 8), mode=Image.BILINEAR):
    """Resize a 256×256 NumPy array to an 8×8 array."""
    im  = Image.fromarray(arr256.astype(np.float32), mode='F')
    imC = im.resize(target_shape[::-1], resample=mode)  # PIL uses (W,H)
    return np.array(imC, dtype=np.float32)

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
            '~/Library/CloudStorage/Box-Box/Hetwet_Data/Synthetic'
        )
        results_dir = Path(os.path.expanduser(base))
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Input dirs
    treatment_dir = results_dir / 'Treatment'
    outcome_dir   = results_dir / 'Outcome'

    # ITE dirs
    ite_dir_direct   = results_dir / 'ITE_Direct'
    ite_dir_direct.mkdir(parents=True, exist_ok=True)
    ite_dir_indirect = results_dir / 'ITE_Indirect'
    ite_dir_indirect.mkdir(parents=True, exist_ok=True)
    ite_dir_total = results_dir / 'ITE_Total'
    ite_dir_total.mkdir(parents=True, exist_ok=True)

    # Theta dir
    theta_dir      = results_dir / 'Theta'
    theta_dir.mkdir(parents=True, exist_ok=True)

    # Spillover Effect dirs
    ite_dir_outgoing    = results_dir / 'ITE_Outgoing'
    ite_dir_outgoing.mkdir(parents=True, exist_ok=True)

    # Outcome dirs
    post_dir      = results_dir / 'Outcome_Post'
    post_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    pdf_ids = set(random.sample(scene_ids, min(5, len(scene_ids))))

    for sid in scene_ids:
        # Paths
        paths = {
            'dem':     Path(folders['dem'])      / f"DEM_{sid}.tiff",
            'cap':     Path(folders['cap'])      / f"CAPITAL_1996_{sid}.tiff",
            'treat':   treatment_dir            / f"treatment_scene_{sid}_gaussian.tiff",
            'actual':  outcome_dir              / f"outcome_scene_{sid}_gaussian.tiff"
        }

        # changing the
        target_shape = (256, 256)
        # Load & resize to WETLAND size
        dem_r    = load_and_resize(paths['dem'],    target_shape, Image.BILINEAR)
        cap_r    = load_and_resize(paths['cap'],    target_shape, Image.BILINEAR)
        treat_r  = load_and_resize(paths['treat'],  target_shape, Image.NEAREST)

        # Load & resize to Claims size
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
        ITE_indirect = upstream_only_KuT(treat, dem, theta, KERNEL)

        # ---------- ITE decomposition ----------
        # 1) direct effect of i on itself
        ITE_direct = theta * treat                    # (H,W)

        # 2) spillover that i produces on *other* pixels -----------------
        ITE_outgoing = downstream_only_KuT(treat, dem, theta, KERNEL)  # (H,W)

        # ---------------------------------------------------------------

        ITE_Total = ITE_direct + ITE_indirect

        # Resize ITES to coarse grid
        ITE_direct   = to_coarse(ITE_direct,   coarse_shape, Image.BILINEAR)
        ITE_indirect = to_coarse(ITE_indirect, coarse_shape, Image.BILINEAR)
        ITE_Total    = to_coarse(ITE_Total,    coarse_shape, Image.BILINEAR)
        ITE_outgoing = to_coarse(ITE_outgoing,    coarse_shape, Image.BILINEAR)
        theta        = to_coarse(theta,        coarse_shape, Image.BILINEAR)

        # save the five rasters
        Image.fromarray(ITE_direct.astype(np.float32), mode='F').save(
            ite_dir_direct / f"ITE_direct_scene_{sid}.tiff")

        Image.fromarray(ITE_indirect.astype(np.float32), mode='F').save(
            ite_dir_indirect / f"ITE_indirect_scene_{sid}.tiff")

        Image.fromarray(ITE_Total.astype(np.float32), mode='F').save(
            ite_dir_total / f"ITE_total_scene_{sid}.tiff")

        Image.fromarray(ITE_outgoing.astype(np.float32), mode='F').save(
            ite_dir_outgoing / f"ITE_Outgoing_scene_{sid}.tiff")

        Image.fromarray(theta.astype(np.float32), mode='F').save(
            theta_dir / f"Theta_scene_{sid}.tiff")
        # ---------- ITE decomposition ----------

        # Post OUTCOME
        post = actual +  ITE_Total

        if noise_type == "gaussian":
            post += np.random.normal(0, noise_sd, size=post.shape)
        elif noise_type != "none":
            raise ValueError("noise_type must be 'gaussian' or 'none'")

        # Save TIFF
        out_tif = post_dir / f"outcome_post_scene_{sid}_{noise_type}.tiff"
        Image.fromarray(post.astype(np.float32), mode='F').save(out_tif)
        logging.info(f"[{sid}] Saved post outcome coarse → {out_tif.name}")

        # PDF compare
        if sid in pdf_ids:
            # 6-panel: Baseline, ITE-direct, ITE-indirect, ITE, Post
            fig, axes = plt.subplots(1, 6, figsize=(16, 8))

            # Baseline
            im0 = axes[0].imshow(actual, cmap='viridis')
            axes[0].set_title('Baseline')
            axes[0].axis('off')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # ITE-direct
            im1 = axes[1].imshow(ITE_direct, cmap='viridis')
            axes[1].set_title('ITE-direct')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # ITE-indirect
            im2 = axes[2].imshow(ITE_indirect, cmap='viridis')
            axes[2].set_title('ITE-indirect')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            # ITE-Total
            im3 = axes[3].imshow(ITE_Total, cmap='viridis')
            axes[3].set_title('ITE-Total')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

            # ITE
            im4 = axes[4].imshow(ITE_outgoing, cmap='viridis')
            axes[4].set_title('ITE Outgoing Effect')
            axes[4].axis('off')
            plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

            # Post
            im5 = axes[5].imshow(post, cmap='viridis')
            axes[5].set_title('Post')
            axes[5].axis('off')
            plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

            fig.suptitle(f"{sid}: Baseline, ITE & Post (coarse {coarse_shape})", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            pdf_path = results_dir / f"{sid}_pre_post_comparison_{noise_type}.pdf"
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"[{sid}] Saved comparison PDF → {pdf_path.name}")

