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


def upstream_only_KuT(T: np.ndarray, DEM: np.ndarray, theta: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Upstream spillover S(i) = Σ_{j: DEM[j] > DEM[i]}  K(d(i,j)) · θ[j] · T[j]

    Parameters
    ----------
    T : np.ndarray          # (H, W)  binary or continuous treatment
    DEM : np.ndarray        # (H, W)  elevations
    theta : np.ndarray      # (H, W)  unit-specific coefficients β(X)
    K : np.ndarray          # (2R+1, 2R+1) distance-decay kernel

    Returns
    -------
    S_up : np.ndarray       # (H, W) upstream-only weighted sum of θ·T
    """
    H, W = T.shape
    R = K.shape[0] // 2

    # Pad all rasters so neighbourhood windows are always full
    T_pad     = np.pad(T,     R, mode='constant', constant_values=0.0)
    theta_pad = np.pad(theta, R, mode='constant', constant_values=0.0)
    DEM_pad   = np.pad(DEM,   R, mode='edge')

    S_up = np.zeros_like(T, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            center_elev = DEM_pad[i + R, j + R]

            # Extract (2R+1)×(2R+1) windows
            dem_patch   = DEM_pad[  i : i + 2*R + 1, j : j + 2*R + 1]
            treat_patch = T_pad[    i : i + 2*R + 1, j : j + 2*R + 1]
            theta_patch = theta_pad[i : i + 2*R + 1, j : j + 2*R + 1]

            # Mask uphill neighbours
            upstream_mask = (dem_patch > center_elev).astype(np.float32)

            # Sum K * θ * T over uphill cells
            S_up[i, j] = np.sum(K * upstream_mask * theta_patch * treat_patch)

    return S_up

def downstream_only_KuT(T: np.ndarray, DEM: np.ndarray, theta: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute the total outgoing ITE that each pixel i exerts on its downhill neighbors:
      S_src(i) = theta[i]*T[i] * sum_{ j: DEM[j] < DEM[i] } K(d(i,j))

    Parameters
    ----------
    T     : np.ndarray  # (H, W)  treatment indicator or intensity
    DEM   : np.ndarray  # (H, W)  elevation field
    theta : np.ndarray  # (H, W)  unit-specific treatment coefficient β(X)
    K     : np.ndarray  # (2R+1, 2R+1) distance-decay kernel

    Returns
    -------
    S_src : np.ndarray  # (H, W) outgoing‐spillover ITE from each pixel
    """
    H, W = T.shape
    R    = K.shape[0] // 2

    # Pad so we can extract full windows at the borders
    T_pad     = np.pad(T,     R, mode='constant', constant_values=0.0)
    theta_pad = np.pad(theta, R, mode='constant', constant_values=0.0)
    DEM_pad   = np.pad(DEM,   R, mode='edge')

    S_src = np.zeros_like(T, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            # 1) read the center pixel's elevation, treatment & theta
            center_elev = DEM_pad[i+R, j+R]
            t_i         = T_pad[i+R, j+R]
            theta_i     = theta_pad[i+R, j+R]

            # 2) neighbourhood of elevations
            dem_patch = DEM_pad[i : i+2*R+1, j : j+2*R+1]

            # 3) downhill mask (which neighbours j have DEM[j] < DEM[i])
            downstream_mask = (dem_patch < center_elev).astype(np.float32)

            # 4) sum up kernel weights over those downhill directions
            weight_sum = np.sum(K * downstream_mask)

            # 5) outgoing ITE = direct‐coefficient×treatment×total downhill weight
            S_src[i, j] = theta_i * t_i * weight_sum

    return S_src
