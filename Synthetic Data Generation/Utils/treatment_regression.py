import os
import logging
from pathlib import Path

import numpy as np
import rasterio as rio
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Utils.load_and_resize import load_and_resize
import matplotlib.pyplot as plt

def logistic_process(
    scene_ids: list[str],
    folders: dict[str, str],
    target_shape: tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    results_dir: str | None = None,
    regularization_C: float = 1.0,
    verbose: bool = True,
    noise_sd: float = 0.2,
    n_trials: int = 1,
    noise_type: str = 'gaussian'
) -> Pipeline:
    """
    Load DEM, capital, and wetness layers for given scene_ids, train a regularized logistic regression,
    save coefficients, and generate synthetic vs actual wetland, DEM, and capital maps for the first scene.

    Parameters
    ----------
    scene_ids : List of scene identifiers to process
    folders : Dict with keys 'dem', 'cap', 'wet' pointing to input directories
    target_shape : Tuple (height, width) to resize each layer
    threshold : Probability cutoff for class map
    results_dir : Directory to save outputs; defaults to Box cloud path
    regularization_C : Inverse of regularization strength for logistic regression
    verbose : If True, print progress messages

    Returns
    -------
    Trained sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Setup results directory
    if results_dir is None:
        base = (
            '~/Library/CloudStorage/Box-Box/Caltech Research/Scripts/'
            'ImageTRT/Synthetic Data Generation/Results'
        )
        results_dir = Path(os.path.expanduser(base))
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dem_list, cap_list, wet_list = [], [], []
    for scene_id in scene_ids:
        if verbose:
            logging.info(f"Processing scene {scene_id}")

        paths = {
            'dem': Path(folders['dem']) / f"DEM_{scene_id}.tiff",
            'cap': Path(folders['cap']) / f"CAPITAL_1996_{scene_id}.tiff",
            'wet': Path(folders['wet']) / f"WETLAND_DEV_1996_2016_{scene_id}.tiff"
        }
        dem_r = load_and_resize(paths['dem'], target_shape, Image.BILINEAR)
        cap_r = load_and_resize(paths['cap'], target_shape, Image.BILINEAR)
        wet_r = load_and_resize(paths['wet'], target_shape, Image.NEAREST)
        if dem_r is None or cap_r is None or wet_r is None:
            continue

        dem_list.append(dem_r.ravel())
        cap_list.append(cap_r.ravel())
        wet_list.append(wet_r.ravel())

    # Prepare design matrix
    X = np.column_stack([np.hstack(dem_list), np.hstack(cap_list)])
    y = np.hstack(wet_list)

    # Build and train pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            penalty='l2',
            C=regularization_C,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
        ))
    ])
    pipe.fit(X, y)

    # Save coefficients
    lr = pipe.named_steps['logreg']
    coefs = np.hstack([lr.intercept_, lr.coef_.flatten()])
    names = ['Intercept', 'DEM', 'Capital']
    coeff_file = results_dir / 'treatment_logistic_coeffs.txt'
    with open(coeff_file, 'w') as f:
        for name, val in zip(names, coefs):
            f.write(f"{name}: {val:.6f}\n")
    if verbose:
        logging.info(f"Coefficients saved to {coeff_file}")

    # Predict for first scene
    scene_id = scene_ids[0]
    dem_arr = dem_list[0]
    cap_arr = cap_list[0]
    wet_arr = wet_list[0]
    X_scene = np.column_stack([dem_arr, cap_arr])
    probs = pipe.predict_proba(X_scene)[:, 1]

    if noise_type.lower() == "gaussian":
    # Add Gaussian jitter, then clip back into [0,1]
        probs = np.clip(
            probs + np.random.normal(loc=0, scale=noise_sd, size=probs.shape), 0, 1
        )
    elif noise_type.lower() == "bernoulli":
        probs = np.random.binomial(n=n_trials, p=probs, size=probs.shape).astype(np.float64)/ n_trials
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian' or 'bernoulli'.")

    # Generate maps
    synthetic_map = (probs.reshape(target_shape) >= threshold).astype(np.uint8)
    actual_map = wet_arr.reshape(target_shape)
    dem_map = dem_arr.reshape(target_shape)
    cap_map = cap_arr.reshape(target_shape)

    # Plot synthetic, actual, DEM, and capital
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # Synthetic wetland development
    axes[0].imshow(synthetic_map, cmap='gray')
    axes[0].set_title(f"Predicted Wetland {scene_id}")
    axes[0].axis('off')

    # Actual wetland development
    axes[1].imshow(actual_map, cmap='gray')
    axes[1].set_title(f"Actual Wetland {scene_id}")
    axes[1].axis('off')

    # DEM
    im2 = axes[2].imshow(dem_map)
    axes[2].set_title(f"DEM {scene_id}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    # Capital
    im3 = axes[3].imshow(cap_map)
    axes[3].set_title(f"Capital {scene_id}")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Scene {scene_id}: Synthetic vs Actual and Inputs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    pdf_path = results_dir / f"scene_{scene_id}_treatment_maps.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    if verbose:
        logging.info(f"Maps saved to {pdf_path}")

    return pipe