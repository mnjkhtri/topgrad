from typing import Union

from .base import Backend
from .numpy_backend import NumpyBackend
from .triton_backend import TritonBackend

_CURRENT_BACKEND: Backend = NumpyBackend()


def get_backend() -> Backend:
    return _CURRENT_BACKEND


def set_backend(backend: Union[str, Backend]) -> Backend:
    """Switch the active backend by name or instance."""
    global _CURRENT_BACKEND
    if isinstance(backend, str):
        name = backend.lower()
        if name == "numpy":
            _CURRENT_BACKEND = NumpyBackend()
        elif name == "triton":
            _CURRENT_BACKEND = TritonBackend()
        else:
            raise ValueError(f"unknown backend '{backend}'")
    elif isinstance(backend, Backend):
        _CURRENT_BACKEND = backend
    else:
        raise TypeError(f"backend must be name or Backend, got {type(backend)}")
    return _CURRENT_BACKEND


def available_backends():
    out = ["numpy"]
    try:
        TritonBackend()
    except Exception:
        pass
    else:
        out.append("triton")
    return out


__all__ = [
    "Backend",
    "NumpyBackend",
    "TritonBackend",
    "available_backends",
    "get_backend",
    "set_backend",
]
