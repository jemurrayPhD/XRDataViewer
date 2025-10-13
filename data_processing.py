
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gauss
    from scipy.ndimage import median_filter as _med
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Processing primitives
# ---------------------------------------------------------------------------

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
        k = int(ksize)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        return _med(Z, size=k, mode="nearest")
    return Z  # no-op fallback

def butterworth(Z: np.ndarray, cutoff: float = 0.1, order: int = 3, btype: str = "low") -> np.ndarray:
    Z = np.asarray(Z, float)
    if not _HAS_SCIPY:
        raise RuntimeError("Butterworth requires SciPy")
    b, a = butter(order, cutoff, btype=btype)
    tmp = filtfilt(b, a, Z, axis=1, padlen=min(3*order, Z.shape[1]-1))
    out = filtfilt(b, a, tmp, axis=0, padlen=min(3*order, Z.shape[0]-1))
    return out


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

    def default_parameters(self) -> Dict[str, float | int | str]:
        return {p.name: p.default for p in self.parameters}

    def apply(self, data: np.ndarray, params: Dict[str, float | int | str]) -> np.ndarray:
        params = dict(self.default_parameters(), **(params or {}))
        return self.func(np.asarray(data, float), **params)


_PROCESSING_FUNCTIONS: Dict[str, ProcessingFunctionSpec] = {
    "gaussian": ProcessingFunctionSpec(
        key="gaussian",
        label="Gaussian blur",
        func=lambda data, sigma=1.0: gaussian_blur(data, sigma=float(sigma)),
        parameters=[
            ParameterDefinition(
                name="sigma",
                label="Sigma",
                kind="float",
                default=1.0,
                minimum=0.1,
                maximum=50.0,
                step=0.1,
            )
        ],
    ),
    "median": ProcessingFunctionSpec(
        key="median",
        label="Median filter",
        func=lambda data, ksize=3: median_filter(data, ksize=int(ksize)),
        parameters=[
            ParameterDefinition(
                name="ksize",
                label="Kernel",
                kind="int",
                default=3,
                minimum=1,
                maximum=99,
                step=2,
            )
        ],
    ),
    "poly": ProcessingFunctionSpec(
        key="poly",
        label="Poly detrend",
        func=lambda data, order_x=2, order_y=2: poly2d_detrend(data, order_x=int(order_x), order_y=int(order_y)),
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
        label="Butterworth",
        func=lambda data, cutoff=0.1, order=3, btype="low": butterworth(data, cutoff=float(cutoff), order=int(order), btype=str(btype)),
        parameters=[
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
        ],
    ),
}


def list_processing_functions() -> List[ProcessingFunctionSpec]:
    return list(_PROCESSING_FUNCTIONS.values())


def get_processing_function(key: str) -> Optional[ProcessingFunctionSpec]:
    return _PROCESSING_FUNCTIONS.get(key)


def apply_processing_step(key: str, data: np.ndarray, params: Optional[Dict[str, float | int | str]] = None) -> np.ndarray:
    spec = get_processing_function(key)
    if spec is None:
        raise KeyError(key)
    return spec.apply(data, params or {})


def summarize_parameters(key: str, params: Dict[str, float | int | str]) -> str:
    spec = get_processing_function(key)
    if spec is None:
        return ""
    parts: List[str] = []
    for p in spec.parameters:
        val = params.get(p.name, p.default)
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


