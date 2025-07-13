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
    Train logistic regression on DEM, Capital, and their scene means to predict wetland development.
    Saves model coefficients and a prediction map for the first scene.

    Returns
    -------
    Trained sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Setup results directory
    if results_dir is None:
        base = (
            '~/Library/CloudStorage/Box-Box/Hetwet_Data/Synthetic'
        )
        results_dir = Path(os.path.expanduser(base))
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists to hold data
    X_list, y_list = [], []

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

        # Flatten
        dem_flat = dem_r.ravel()
        cap_flat = cap_r.ravel()
        wet_flat = wet_r.ravel()

        # Scene-level means
        # dem_mean = np.mean(dem_flat)
        # cap_mean = np.mean(cap_flat)

        # Repeat means
        # dem_mean_col = np.full_like(dem_flat, dem_mean)
        # cap_mean_col = np.full_like(cap_flat, cap_mean)

        # Full feature matrix
        # X_scene = np.column_stack([dem_flat, dem_mean_col, cap_flat, cap_mean_col])
        X_scene = np.column_stack([dem_flat, cap_flat])
        X_list.append(X_scene)
        y_list.append(wet_flat)

    if not X_list:
        raise RuntimeError("No valid scenes found. Check file paths and inputs.")

    # Final design matrix
    X = np.vstack(X_list)
    y = np.hstack(y_list)

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
    # names = ['Intercept', 'DEM', 'DEM_Mean', 'Capital', 'Capital_Mean']
    names = ['Intercept', 'DEM', 'Capital']
    coeff_file = results_dir / 'treatment_logistic_coeffs.txt'
    with open(coeff_file, 'w') as f:
        for name, val in zip(names, coefs):
            f.write(f"{name}: {val:.6f}\n")
    if verbose:
        logging.info(f"Coefficients saved to {coeff_file}")

    # Predict for first scene
    scene_id = scene_ids[0]
    X_scene = X_list[0]
    dem_arr = X_scene[:, 0]
    cap_arr = X_scene[:, 1]
    wet_arr = y_list[0]

    probs = pipe.predict_proba(X_scene)[:, 1]

    # Add noise
    if noise_type.lower() == "gaussian":
        probs = np.clip(probs + np.random.normal(0, noise_sd, size=probs.shape), 0, 1)
    elif noise_type.lower() == "bernoulli":
        probs = np.random.binomial(n=n_trials, p=probs, size=probs.shape).astype(np.float64) / n_trials
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Generate maps
    synthetic_map = (probs.reshape(target_shape) >= threshold).astype(np.uint8)
    actual_map = wet_arr.reshape(target_shape)
    dem_map = dem_arr.reshape(target_shape)
    cap_map = cap_arr.reshape(target_shape)

    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(synthetic_map, cmap='gray')
    axes[0].set_title(f"Predicted Wetland {scene_id}")
    axes[0].axis('off')

    axes[1].imshow(actual_map, cmap='gray')
    axes[1].set_title(f"Actual Wetland {scene_id}")
    axes[1].axis('off')

    im2 = axes[2].imshow(dem_map)
    axes[2].set_title(f"DEM {scene_id}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(cap_map)
    axes[3].set_title(f"Capital {scene_id}")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"Scene {scene_id}: Synthetic vs Actual and Inputs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = results_dir / f"scene_{scene_id}_treatment_maps.pdf"
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    if verbose:
        logging.info(f"Maps saved to {pdf_path}")

    return pipe
