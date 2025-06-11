import os
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from Utils.load_and_resize import load_and_resize
import matplotlib.pyplot as plt

def regression_process(
    scene_ids: list[str],
    folders: dict[str, str],
    target_shape: tuple[int, int] = (8, 8),
    results_dir: str | None = None,
    verbose: bool = True,
    noise_type: str = 'gaussian',      # 'gaussian' or 'none'
    noise_sd: float = 0.2              # only used if noise_type=='gaussian'
) -> Pipeline:
    """
    1) Loads DEM, capital, and two log-claims rasters (1996 & 2016) per scene.
    2) Computes log_outcome = claims_16 - claims_96.
    3) Fits a linear regression: log_outcome ~ DEM + Capital.
    4) Saves coefficients.
    5) For the first scene, predicts (with optional Gaussian noise), and plots:
       Predicted vs Actual continuous map, plus DEM & Capital inputs.

    Returns
    -------
    Trained sklearn Pipeline (StandardScaler + LinearRegression).
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ——— Setup results dir ———
    if results_dir is None:
        base = (
            '~/Library/CloudStorage/Box-Box/Caltech Research/Scripts/'
            'ImageTRT/Synthetic Data Generation/Results'
        )
        results_dir = Path(os.path.expanduser(base))
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ——— Load & stack data ———
    dem_list, cap_list, outcome_list = [], [], []
    for sid in scene_ids:
        if verbose:
            logging.info(f"Loading scene {sid}")

        # build paths
        paths = {
            'dem':       Path(folders['dem'])       / f"DEM_{sid}.tiff",
            'cap':       Path(folders['cap'])       / f"CAPITAL_1996_{sid}.tiff",
            'claims96':  Path(folders['claims_96']) / f"LOG_CLAIMS_1996_{sid}.tiff",
            'claims16':  Path(folders['claims_16']) / f"LOG_CLAIMS_2016_{sid}.tiff",
        }

        # resize rasters
        dem_r      = load_and_resize(paths['dem'],      target_shape, Image.BILINEAR)
        cap_r      = load_and_resize(paths['cap'],      target_shape, Image.BILINEAR)
        claims96_r = load_and_resize(paths['claims96'], target_shape, Image.NEAREST)
        claims16_r = load_and_resize(paths['claims16'], target_shape, Image.NEAREST)
        if dem_r is None or cap_r is None or claims96_r is None or claims16_r is None:
            logging.warning(f"Skipping {sid}: failed to load all inputs.")
            continue

        # flatten & compute outcome
        dem_list.append(dem_r.ravel())
        cap_list.append(cap_r.ravel())
        log_outcome = (claims16_r - claims96_r).ravel()
        outcome_list.append(log_outcome)

    # design matrix & target
    X = np.column_stack([np.hstack(dem_list), np.hstack(cap_list)])
    y = np.hstack(outcome_list)

    # ——— Build & train regression pipeline ———
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('linreg', LinearRegression())
    ])
    pipe.fit(X, y)

    # ——— Save coefficients ———
    lr = pipe.named_steps['linreg']
    coefs = np.hstack([lr.intercept_, lr.coef_])
    names = ['Intercept', 'DEM', 'Capital']
    coeff_file = results_dir / 'outcome_regression_coeffs.txt'
    with open(coeff_file, 'w') as f:
        for name, val in zip(names, coefs):
            f.write(f"{name}: {val:.6f}\n")
    if verbose:
        logging.info(f"Coefficients saved to {coeff_file}")

    # ——— Predict & Plot for first scene ———
    sid = scene_ids[0]
    dem_arr = dem_list[0]
    cap_arr = cap_list[0]
    actual_out = outcome_list[0]

    X_scene = np.column_stack([dem_arr, cap_arr])
    pred = pipe.predict(X_scene)

    # add optional Gaussian noise
    if noise_type.lower() == 'gaussian':
        pred = pred + np.random.normal(0, noise_sd, size=pred.shape)
    elif noise_type.lower() != 'none':
        raise ValueError("noise_type must be 'gaussian' or 'none'")

    # reshape back to 2D maps
    pred_map   = pred.reshape(target_shape)
    actual_map = actual_out.reshape(target_shape)
    dem_map    = dem_arr.reshape(target_shape)
    cap_map    = cap_arr.reshape(target_shape)

    # plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    im0 = axes[0].imshow(pred_map)
    axes[0].set_title(f"Predicted ΔLogClaims {sid}")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(actual_map)
    axes[1].set_title(f"Actual ΔLogClaims {sid}")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(dem_map)
    axes[2].set_title(f"DEM {sid}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(cap_map)
    axes[3].set_title(f"Capital {sid}")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Scene {sid}: Continuous Outcome vs Inputs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = results_dir / f"scene_{sid}_continuous_outcome.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    if verbose:
        logging.info(f"Maps saved to {pdf_path}")

    return pipe
