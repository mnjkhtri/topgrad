from typing import Union

from .numpy_ops import NumpyBackend

try:  # pragma: no cover - triton is optional
    from .triton_ops import TritonBackend
except Exception:  # pragma: no cover - keep numpy-only environments working
    TritonBackend = None

_CURRENT_BACKEND = NumpyBackend()


def get_backend():
    return _CURRENT_BACKEND


def set_backend(backend: Union[str, NumpyBackend]):
    """Switch the active backend by name or instance."""
    global _CURRENT_BACKEND
    if isinstance(backend, str):
        name = backend.lower()
        if name == "numpy":
            _CURRENT_BACKEND = NumpyBackend()
        elif name == "triton":
            if TritonBackend is None:
                raise ImportError("Triton backend is unavailable; install triton.")
            _CURRENT_BACKEND = TritonBackend()
        else:
            raise ValueError(f"unknown backend '{backend}'")
    elif isinstance(backend, NumpyBackend) or (
        TritonBackend is not None and isinstance(backend, TritonBackend)
    ):
        _CURRENT_BACKEND = backend
    else:
        raise TypeError(
            f"backend must be name or NumpyBackend/TritonBackend, got {type(backend)}"
        )
    return _CURRENT_BACKEND


def available_backends():
    backends = ["numpy"]
    if TritonBackend is not None:
        backends.append("triton")
    return backends


__all__ = [
    "NumpyBackend",
    "TritonBackend",
    "available_backends",
    "get_backend",
    "set_backend",
]
