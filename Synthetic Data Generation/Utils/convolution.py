import numpy as np
from scipy.signal import fftconvolve

def make_exp_kernel(lam_m, cell_size_m, truncate=4):
    """
    Build a normalized 2D exponential‐decay kernel:
      K(d) = exp(−d/lam_m), truncated at 4·lam_m.
    """
    R = int(truncate * lam_m / cell_size_m)
    y, x = np.ogrid[-R:R+1, -R:R+1]
    d    = np.hypot(x, y) * cell_size_m
    K    = np.exp(-d/lam_m)
    K   /= K.sum()
    return K


def exp_convolve(arr: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convolve a 2D array `arr` with kernel `K` using FFT, returning an array of the same shape.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array to be convolved.
    K : np.ndarray
        2D kernel array.

    Returns
    -------
    out : np.ndarray
        Convolved array, same shape as `arr`.
    """
    # Ensure float32 for speed and precision
    return fftconvolve(arr.astype(np.float32), K.astype(np.float32), mode='same')


def upstream_only_KuT(T: np.ndarray, DEM: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute upstream-only convolution KuT_up(x) = sum_{j: DEM[j] > DEM[x]} K(d(x,j)) * T[j]

    Parameters
    ----------
    T : np.ndarray
        2D treatment array (e.g., synthetic wetland) shape (H, W).
    DEM : np.ndarray
        2D digital elevation model array shape (H, W).
    K : np.ndarray
        Precomputed exponential kernel shape (2R+1, 2R+1).

    Returns
    -------
    KuT_up : np.ndarray
        2D array of upstream-only weighted sum, same shape as `T`.
    """
    H, W = T.shape
    R = K.shape[0] // 2

    # Pad arrays to handle borders
    T_pad = np.pad(T, pad_width=R, mode='constant', constant_values=0.0)
    DEM_pad = np.pad(DEM, pad_width=R, mode='edge')

    # Precompute coordinate offsets mask
    i_idxs, j_idxs = np.indices(K.shape)
    # Upstream convolution result
    KuT_up = np.zeros_like(T, dtype=np.float32)

    # Loop over each pixel
    for i in range(H):
        for j in range(W):
            center_elev = DEM_pad[i + R, j + R]
            dem_patch = DEM_pad[i : i + 2*R + 1, j : j + 2*R + 1]
            t_patch = T_pad[i : i + 2*R + 1, j : j + 2*R + 1]
            # Mask where neighbor elevation is strictly greater
            upstream_mask = (dem_patch > center_elev).astype(np.float32)
            # Weighted sum
            KuT_up[i, j] = np.sum(K * upstream_mask * t_patch)

    return KuT_up