import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from typing import List, Dict, Tuple, Union


def compute_pca_components(
    emb_source: Union[str, pd.DataFrame],
    n_components: int = 10
) -> pd.DataFrame:
    """
    Load an embeddings table (scene_id + emb_0…emb_D) and return the first
    `n_components` principal‐component scores for each scene.

    Parameters
    ----------
    emb_source : str or pd.DataFrame
        Either the path to a CSV (with columns 'scene_id','emb_0',…) or
        a DataFrame already loaded.
    n_components : int
        How many principal components to keep (default 10).

    Returns
    -------
    pd.DataFrame
        A DataFrame of shape (n_scenes, n_components+1) with columns
        ['scene_id', 'PC1', …, f'PC{n_components}'].
    """
    # load if necessary
    if isinstance(emb_source, str):
        df = pd.read_csv(emb_source)
    else:
        df = emb_source.copy()

    # peel off scene IDs and the raw embedding matrix
    scene_ids = df['scene_id']
    X = df.drop(columns='scene_id').values

    # run PCA
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)

    # build result frame
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    result = pd.DataFrame(pcs, columns=pc_cols)
    result.insert(0, 'scene_id', scene_ids.values)

    return result


def get_train_test_indices(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train‐ and test‐index arrays for a single split of df.
    """
    train_idx, test_idx = train_test_split(
        df.index.values, test_size=test_size, random_state=random_state
    )
    return train_idx, test_idx


def estimate_treatment_effect_tabular(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outcome_col: str,
    treatment_col: str,
    covariate_cols: List[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], CausalForestDML]:
    """
    Use the given train/test indices, fit a CausalForestDML on scalar T,
    and return:
      - cate_df: [scene_id, dataset(train/test), CATE]
      - ate_scores: {'ate_train':…, 'ate_test':…}
      - the fitted model
    """
    # slice
    train = df.loc[train_idx]
    test  = df.loc[test_idx]

    # assemble arrays
    X_tr = train[covariate_cols].values
    T_tr = train[treatment_col].values
    Y_tr = train[outcome_col].values

    X_te = test [covariate_cols].values
    T_te = test [treatment_col].values
    Y_te = test [outcome_col].values

    # fit
    cf = CausalForestDML(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    cf.fit(Y_tr, T_tr, X=X_tr)

    # predict
    cate_tr = cf.effect(X_tr)
    cate_te = cf.effect(X_te)

    # ATE
    ate_tr = cf.ate(X_tr)
    ate_te = cf.ate(X_te)

    # build output df
    df_tr = pd.DataFrame({
        'scene_id': train['scene_id'].values,
        'dataset': 'train',
        'CATE': cate_tr
    })
    df_te = pd.DataFrame({
        'scene_id': test ['scene_id'].values,
        'dataset': 'test',
        'CATE': cate_te
    })

    return pd.concat([df_tr, df_te], ignore_index=True), \
           {'ate_train': ate_tr, 'ate_test': ate_te}, \
           cf


from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML

def estimate_treatment_effect_with_embeddings(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outcome_col: str,
    treatment_col: List[str],   # PCA columns
    covariate_cols: List[str],
    n_estimators: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float], CausalForestDML]:
    """
    Estimate treatment effects using image PCA embeddings as multivariate treatments.
    Returns per-sample CATE estimates and train/test ATE.
    """
    # Prepare
    df2 = df.copy()
    df2['T_vec'] = df2[treatment_col].values.tolist()

    train = df2.loc[train_idx]
    test  = df2.loc[test_idx]

    X_tr = train[covariate_cols].values
    T_tr = np.vstack(train['T_vec'].values)
    Y_tr = train[outcome_col].values

    X_te = test[covariate_cols].values
    T_te = np.vstack(test['T_vec'].values)
    Y_te = test[outcome_col].values

    # Fit model
    cf = CausalForestDML(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    cf.fit(Y_tr, T_tr, X=X_tr)

    # Predict CATEs
    cate_tr = cf.effect(X_tr)
    cate_te = cf.effect(X_te)

    # Predict ATEs using proper baseline for multivariate treatment
    baseline = np.zeros(T_tr.shape[1])
    ate_tr = cf.ate(X_tr, T0=baseline)
    ate_te = cf.ate(X_te, T0=baseline)

    # Collect results
    df_tr = pd.DataFrame({
        'scene_id': train['scene_id'].values,
        'dataset': 'train',
        'CATE_wet': cate_tr
    })
    df_te = pd.DataFrame({
        'scene_id': test['scene_id'].values,
        'dataset': 'test',
        'CATE_wet': cate_te
    })

    cate_df = pd.concat([df_tr, df_te], ignore_index=True)
    ate_dict = {'ate_train': ate_tr, 'ate_test': ate_te}

    return cate_df, ate_dict, cf


import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def evaluate_ite(
    df_actual: pd.DataFrame,
    df_estimated: pd.DataFrame,
    id_col: str = "scene_id",
    actual_col: str = "actual_ite",
    estimated_col: str = "estimated_ite",
    output_dir: str = ".",
) -> dict:
    """
    Merge actual and estimated ITE DataFrames, compute R², RMSE, RATE ratio,
    plot actual vs. estimated ITE, and plot & save histograms of both.

    Parameters
    ----------
    df_actual : pd.DataFrame
        Must contain [id_col, actual_col].
    df_estimated : pd.DataFrame
        Must contain [id_col, estimated_col].
    id_col : str
        Column name to join on. Default "scene_id".
    actual_col : str
        Column name for the true ITE values. Default "actual_ite".
    estimated_col : str
        Column name for the estimated ITE values. Default "estimated_ite".
    output_dir : str
        Directory to save histogram PNGs. Will be created if it doesn't exist.

    Returns
    -------
    metrics : dict
        {
            "r2": float,           # Coefficient of determination
            "rmse": float,         # Root mean squared error
            "rate_ratio": float    # mean(estimated_ite)/mean(actual_ite)
        }

    Side effects
    ------------
    - Displays a scatter plot of actual vs. estimated ITE with the 45° line.
    - Saves two histogram plots:
        <output_dir>/hist_actual_ite.png
        <output_dir>/hist_estimated_ite.png
    """
    # 1) Merge on scene_id
    df = pd.merge(
        df_actual[[id_col, actual_col]],
        df_estimated[[id_col, estimated_col]],
        on=id_col,
        how="inner",
    )

    # 2) Extract arrays
    y_true = df[actual_col].to_numpy()
    y_pred = df[estimated_col].to_numpy()

    # 3) Compute metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ate_true = y_true.mean()
    ate_pred = y_pred.mean()
    rate_ratio = ate_pred / ate_true if ate_true != 0 else np.nan

    # 4) Scatter plot actual vs. estimated
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Actual ITE")
    plt.ylabel("Estimated ITE")
    plt.title(
        f"ITE Estimates\nR² = {r2:.3f}, RMSE = {rmse:.3f}, RATE ratio = {rate_ratio:.3f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 6) Plot & save histogram of actual ITE
    plt.figure()
    plt.hist(y_true, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Actual ITE")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actual ITE")
    plt.tight_layout()
    actual_path = os.path.join(output_dir, "hist_actual_ite.png")
    plt.savefig(actual_path)
    plt.close()

    # 7) Plot & save histogram of estimated ITE
    plt.figure()
    plt.hist(y_pred, bins=30, alpha=0.7, edgecolor="black")
    plt.xlabel("Estimated ITE")
    plt.ylabel("Frequency")
    plt.title("Distribution of Estimated ITE")
    plt.tight_layout()
    est_path = os.path.join(output_dir, "hist_estimated_ite.png")
    plt.savefig(est_path)
    plt.close()

    print(f"Saved histograms to:\n  {actual_path}\n  {est_path}")

    return {"r2": r2, "rmse": rmse, "rate_ratio": rate_ratio}

