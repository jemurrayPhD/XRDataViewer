
from __future__ import annotations
import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gauss
    from scipy.ndimage import median_filter as _med
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def poly2d_detrend(Z: np.ndarray, order_x: int = 2, order_y: int = 2) -> np.ndarray:
    Z = np.asarray(Z, float)
    Ny, Nx = Z.shape
    y, x = np.mgrid[0:Ny, 0:Nx]
    terms = []
    for oy in range(order_y + 1):
        for ox in range(order_x + 1):
            terms.append((x**ox) * (y**oy))
    A = np.vstack([t.ravel() for t in terms]).T
    coef, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    fit = (A @ coef).reshape(Ny, Nx)
    return Z - fit

def gaussian_blur(Z: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    Z = np.asarray(Z, float)
    if _HAS_SCIPY:
        return _gauss(Z, sigma=float(sigma), mode="nearest")
    return Z  # no-op fallback

def median_filter(Z: np.ndarray, ksize: int = 3) -> np.ndarray:
    Z = np.asarray(Z, float)
    if _HAS_SCIPY:
        return _med(Z, size=int(ksize), mode="nearest")
    return Z  # no-op fallback

def butterworth(Z: np.ndarray, cutoff: float = 0.1, order: int = 3, btype: str = "low") -> np.ndarray:
    Z = np.asarray(Z, float)
    if not _HAS_SCIPY:
        raise RuntimeError("Butterworth requires SciPy")
    b, a = butter(order, cutoff, btype=btype)
    tmp = filtfilt(b, a, Z, axis=1, padlen=min(3*order, Z.shape[1]-1))
    out = filtfilt(b, a, tmp, axis=0, padlen=min(3*order, Z.shape[0]-1))
    return out
