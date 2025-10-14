
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gauss
    from scipy.ndimage import median_filter as _med
    from scipy.signal import butter, filtfilt, savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    savgol_filter = None  # type: ignore


# ---------------------------------------------------------------------------
# Processing primitives
# ---------------------------------------------------------------------------

def poly2d_detrend(Z: np.ndarray, order_x: int = 2, order_y: int = 2) -> np.ndarray:
    """Subtract a 2D polynomial fit from *Z* to remove broad background trends."""

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

def _axis_choices(dims: Optional[Sequence[str]]) -> List[tuple[str, str]]:
    """Return axis selection labels used by anisotropic filter parameter factories."""

    labels = []
    if not dims:
        return labels
    for idx, name in enumerate(dims):
        label = name if name else f"Axis {idx}"
        labels.append((f"Axis: {label}", f"axis:{idx}"))
    return labels


def gaussian_filter_nd(
    data: np.ndarray, sigma: float = 1.0, mode: str = "iso"
) -> np.ndarray:
    """Apply an N-dimensional Gaussian filter with isotropic or axis-wise control."""

    data = np.asarray(data, float)
    if not _HAS_SCIPY:
        return data
    sigma = float(sigma)
    ndim = data.ndim
    if ndim <= 1:
        return _gauss(data, sigma=sigma, mode="nearest")
    if mode == "iso":
        return _gauss(data, sigma=sigma, mode="nearest")
    if mode == "axis:all":
        result = data
        for axis in range(ndim):
            sigmas = [0.0] * ndim
            sigmas[axis] = sigma
            result = _gauss(result, sigma=sigmas, mode="nearest")
        return result
    if mode.startswith("axis:"):
        try:
            axis = int(mode.split(":", 1)[1])
        except Exception:
            axis = 0
        axis = max(0, min(ndim - 1, axis))
        sigmas = [0.0] * ndim
        sigmas[axis] = sigma
        return _gauss(data, sigma=sigmas, mode="nearest")
    return _gauss(data, sigma=sigma, mode="nearest")


def median_filter_nd(data: np.ndarray, ksize: int = 3, mode: str = "iso") -> np.ndarray:
    """Run a median filter either isotropically or along individual axes."""

    data = np.asarray(data, float)
    if not _HAS_SCIPY:
        return data
    k = int(ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if data.ndim <= 1 or mode == "iso":
        return _med(data, size=k, mode="nearest")
    if mode == "axis:all":
        result = data
        for axis in range(data.ndim):
            size = [1] * data.ndim
            size[axis] = k
            result = _med(result, size=tuple(size), mode="nearest")
        return result
    if mode.startswith("axis:"):
        try:
            axis = int(mode.split(":", 1)[1])
        except Exception:
            axis = 0
        axis = max(0, min(data.ndim - 1, axis))
        size = [1] * data.ndim
        size[axis] = k
        return _med(data, size=tuple(size), mode="nearest")
    return _med(data, size=k, mode="nearest")


def butterworth_filter(
    data: np.ndarray,
    cutoff: float = 0.1,
    order: int = 3,
    btype: str = "low",
    mode: str = "iso",
) -> np.ndarray:
    """Apply a Butterworth filter, optionally sweeping each axis of an array."""

    data = np.asarray(data, float)
    if not _HAS_SCIPY:
        raise RuntimeError("Butterworth requires SciPy")
    b, a = butter(order, cutoff, btype=btype)

    def _apply_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        axis = max(0, min(arr.ndim - 1, axis))
        length = arr.shape[axis]
        if length <= 1:
            return arr
        padlen = min(max(1, 3 * order), length - 1)
        try:
            return filtfilt(b, a, arr, axis=axis, padlen=padlen)
        except Exception:
            return filtfilt(b, a, arr, axis=axis, padlen=max(0, length - 1))

    ndim = data.ndim
    if ndim <= 1:
        return _apply_axis(data, 0)
    if mode == "iso":
        result = data
        for axis in range(ndim):
            result = _apply_axis(result, axis)
        return result
    if mode == "axis:all":
        result = data
        for axis in range(ndim):
            result = _apply_axis(result, axis)
        return result
    if mode.startswith("axis:"):
        try:
            axis = int(mode.split(":", 1)[1])
        except Exception:
            axis = 0
        return _apply_axis(data, axis)
    return _apply_axis(data, 0)


def poly1d_detrend(data: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove a low-order polynomial baseline from 1D data."""

    arr = np.asarray(data, float).reshape(-1)
    if arr.size == 0:
        return arr
    x = np.arange(arr.size, dtype=float)
    order = max(0, int(order))
    try:
        coeffs = np.polyfit(x, arr, order)
        fit = np.polyval(coeffs, x)
    except Exception:
        return arr
    return arr - fit


def moving_average_1d(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 1D signal with a centered moving-average kernel."""

    arr = np.asarray(data, float).reshape(-1)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def savgol_1d(data: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Perform Savitzky–Golay filtering on 1D data when SciPy is available."""

    arr = np.asarray(data, float).reshape(-1)
    if not _HAS_SCIPY or savgol_filter is None:
        return arr
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    poly = max(0, min(int(poly), window - 1))
    try:
        return savgol_filter(arr, window_length=window, polyorder=poly, mode="interp")
    except Exception:
        return arr


# ---------------------------------------------------------------------------
# Declarative processing definitions
# ---------------------------------------------------------------------------


@dataclass
class ParameterDefinition:
    name: str
    label: str
    kind: str
    default: float | int | str
    minimum: Optional[float | int] = None
    maximum: Optional[float | int] = None
    step: Optional[float | int] = None
    choices: Optional[List[tuple[str, str]]] = None


@dataclass
class ProcessingFunctionSpec:
    key: str
    label: str
    func: Callable[..., np.ndarray]
    parameters: Iterable[ParameterDefinition] = field(default_factory=list)
    min_ndim: int = 1
    max_ndim: Optional[int] = None
    parameter_factory: Optional[Callable[[Optional[Sequence[str]]], Iterable[ParameterDefinition]]] = None

    def supports_ndim(self, ndim: Optional[int]) -> bool:
        if ndim is None:
            return True
        if ndim < self.min_ndim:
            return False
        if self.max_ndim is not None and ndim > self.max_ndim:
            return False
        return True

    def parameters_for_data(self, dims: Optional[Sequence[str]]) -> List[ParameterDefinition]:
        if self.parameter_factory:
            return list(self.parameter_factory(dims))
        return list(self.parameters)

    def default_parameters(self, dims: Optional[Sequence[str]] = None) -> Dict[str, float | int | str]:
        return {p.name: p.default for p in self.parameters_for_data(dims)}

    def apply(
        self,
        data: np.ndarray,
        params: Dict[str, float | int | str],
        dims: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        array = np.asarray(data, float)
        defaults = self.default_parameters(dims)
        merged = dict(defaults, **(params or {}))
        return self.func(array, **merged)


def _gaussian_parameters(dims: Optional[Sequence[str]]) -> List[ParameterDefinition]:
    """Build parameter definitions for Gaussian filtering based on dimensionality."""

    params = [
        ParameterDefinition(
            name="sigma",
            label="Sigma",
            kind="float",
            default=1.0,
            minimum=0.1,
            maximum=50.0,
            step=0.1,
        )
    ]
    if dims is None or len(dims) > 1:
        choices = [("Isotropic", "iso"), ("Each axis (sequential)", "axis:all")]
        if dims and len(dims) > 1:
            choices.extend(_axis_choices(dims))
        params.append(
            ParameterDefinition(
                name="mode",
                label="Mode",
                kind="enum",
                default="iso",
                choices=choices,
            )
        )
    return params


def _median_parameters(dims: Optional[Sequence[str]]) -> List[ParameterDefinition]:
    """Create the parameter set for median filters, adapting axis choices."""

    params = [
        ParameterDefinition(
            name="ksize",
            label="Kernel",
            kind="int",
            default=3,
            minimum=1,
            maximum=99,
            step=2,
        )
    ]
    if dims is None or len(dims) > 1:
        choices = [("Isotropic", "iso"), ("Each axis (sequential)", "axis:all")]
        if dims and len(dims) > 1:
            choices.extend(_axis_choices(dims))
        params.append(
            ParameterDefinition(
                name="mode",
                label="Mode",
                kind="enum",
                default="iso",
                choices=choices,
            )
        )
    return params


def _butter_parameters(dims: Optional[Sequence[str]]) -> List[ParameterDefinition]:
    """Return Butterworth parameter definitions including axis handling options."""

    params = [
        ParameterDefinition(
            name="cutoff",
            label="Cutoff",
            kind="float",
            default=0.1,
            minimum=0.001,
            maximum=0.5,
            step=0.01,
        ),
        ParameterDefinition(
            name="order",
            label="Order",
            kind="int",
            default=3,
            minimum=1,
            maximum=10,
            step=1,
        ),
        ParameterDefinition(
            name="btype",
            label="Type",
            kind="enum",
            default="low",
            choices=[("Low pass", "low"), ("High pass", "high"), ("Band pass", "bandpass")],
        ),
    ]
    if dims is None or len(dims) > 1:
        choices = [("Isotropic", "iso"), ("Each axis (sequential)", "axis:all")]
        if dims and len(dims) > 1:
            choices.extend(_axis_choices(dims))
        params.append(
            ParameterDefinition(
                name="mode",
                label="Mode",
                kind="enum",
                default="iso",
                choices=choices,
            )
        )
    return params


_PROCESSING_FUNCTIONS: Dict[str, ProcessingFunctionSpec] = {
    "gaussian": ProcessingFunctionSpec(
        key="gaussian",
        label="Gaussian smoothing",
        func=lambda data, sigma=1.0, mode="iso": gaussian_filter_nd(
            data, sigma=float(sigma), mode=str(mode)
        ),
        parameter_factory=_gaussian_parameters,
    ),
    "median": ProcessingFunctionSpec(
        key="median",
        label="Median filter",
        func=lambda data, ksize=3, mode="iso": median_filter_nd(
            data, ksize=int(ksize), mode=str(mode)
        ),
        parameter_factory=_median_parameters,
    ),
    "poly": ProcessingFunctionSpec(
        key="poly",
        label="Polynomial detrend (2D)",
        func=lambda data, order_x=2, order_y=2: poly2d_detrend(
            data, order_x=int(order_x), order_y=int(order_y)
        ),
        min_ndim=2,
        parameters=[
            ParameterDefinition(
                name="order_x",
                label="Order X",
                kind="int",
                default=2,
                minimum=0,
                maximum=6,
                step=1,
            ),
            ParameterDefinition(
                name="order_y",
                label="Order Y",
                kind="int",
                default=2,
                minimum=0,
                maximum=6,
                step=1,
            ),
        ],
    ),
    "butterworth": ProcessingFunctionSpec(
        key="butterworth",
        label="Butterworth filter",
        func=lambda data, cutoff=0.1, order=3, btype="low", mode="iso": butterworth_filter(
            data,
            cutoff=float(cutoff),
            order=int(order),
            btype=str(btype),
            mode=str(mode),
        ),
        parameter_factory=_butter_parameters,
    ),
    "poly1d": ProcessingFunctionSpec(
        key="poly1d",
        label="Polynomial baseline (1D)",
        func=lambda data, order=2: poly1d_detrend(data, order=int(order)),
        min_ndim=1,
        max_ndim=1,
        parameters=[
            ParameterDefinition(
                name="order",
                label="Order",
                kind="int",
                default=2,
                minimum=0,
                maximum=10,
                step=1,
            )
        ],
    ),
    "moving_average": ProcessingFunctionSpec(
        key="moving_average",
        label="Moving average (1D)",
        func=lambda data, window=5: moving_average_1d(data, window=int(window)),
        min_ndim=1,
        max_ndim=1,
        parameters=[
            ParameterDefinition(
                name="window",
                label="Window",
                kind="int",
                default=5,
                minimum=1,
                maximum=501,
                step=2,
            )
        ],
    ),
    "savgol": ProcessingFunctionSpec(
        key="savgol",
        label="Savitzky-Golay (1D)",
        func=lambda data, window=5, poly=2: savgol_1d(data, window=int(window), poly=int(poly)),
        min_ndim=1,
        max_ndim=1,
        parameters=[
            ParameterDefinition(
                name="window",
                label="Window",
                kind="int",
                default=7,
                minimum=3,
                maximum=301,
                step=2,
            ),
            ParameterDefinition(
                name="poly",
                label="Polynomial",
                kind="int",
                default=2,
                minimum=0,
                maximum=10,
                step=1,
            ),
        ],
    ),
}


def list_processing_functions(ndim: Optional[int] = None) -> List[ProcessingFunctionSpec]:
    """Enumerate processing functions that support arrays of the requested dimension."""

    return [spec for spec in _PROCESSING_FUNCTIONS.values() if spec.supports_ndim(ndim)]


def get_processing_function(key: str) -> Optional[ProcessingFunctionSpec]:
    """Look up a processing function specification by registry key."""

    return _PROCESSING_FUNCTIONS.get(key)


def apply_processing_step(
    key: str,
    data: np.ndarray,
    params: Optional[Dict[str, float | int | str]] = None,
) -> np.ndarray:
    """Execute the named processing step using *params* on *data*."""

    spec = get_processing_function(key)
    if spec is None:
        raise KeyError(key)
    array = np.asarray(data, float)
    dims = tuple(f"axis{i}" for i in range(array.ndim))
    return spec.apply(array, params or {}, dims=dims)


def summarize_parameters(key: str, params: Dict[str, float | int | str]) -> str:
    """Produce a compact human-readable summary for a processing configuration."""

    spec = get_processing_function(key)
    if spec is None:
        return ""
    parts: List[str] = []
    definitions = spec.parameters_for_data(None)
    for p in definitions:
        val = params.get(p.name, p.default)
        if p.kind == "enum" and p.choices:
            label_map = {value: label for label, value in p.choices}
            val = label_map.get(val, val)
        parts.append(f"{p.label}={val}")
    return ", ".join(parts)


@dataclass
class ProcessingStep:
    key: str
    params: Dict[str, float | int | str]

    def to_dict(self) -> Dict[str, object]:
        return {"key": self.key, "params": dict(self.params)}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ProcessingStep":
        key = str(data.get("key"))
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Invalid params for ProcessingStep")
        return ProcessingStep(key=key, params=dict(params))


@dataclass
class ProcessingPipeline:
    name: str
    steps: List[ProcessingStep] = field(default_factory=list)

    def apply(self, data: np.ndarray) -> np.ndarray:
        result = np.asarray(data, float)
        for step in self.steps:
            result = apply_processing_step(step.key, result, step.params)
        return result

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "ProcessingPipeline":
        name = str(data.get("name", ""))
        steps_data = data.get("steps") or []
        if not isinstance(steps_data, list):
            raise ValueError("Invalid steps for ProcessingPipeline")
        steps = [ProcessingStep.from_dict(sd) for sd in steps_data]
        return ProcessingPipeline(name=name, steps=steps)


