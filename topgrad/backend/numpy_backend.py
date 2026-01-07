import numpy as np

from .base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self):
        super().__init__()
        self.xp = np

    def is_array(self, x):
        return isinstance(x, np.ndarray)

    def asarray(self, data, dtype=None):
        dtype = dtype or self.float32
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        dtype = dtype or self.float32
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        dtype = dtype or self.float32
        return np.ones(shape, dtype=dtype)

    def ones_like(self, x):
        return np.ones_like(x)

    def full(self, shape, fill_value, dtype=None):
        dtype = dtype or getattr(fill_value, "dtype", self.float32)
        return np.full(shape, fill_value, dtype=dtype)

    def randn(self, shape):
        return np.random.standard_normal(shape).astype(self.float32)

    def empty(self, shape, dtype=None):
        dtype = dtype or self.float32
        return np.empty(shape, dtype=dtype)

    def pad(self, x, pad_width, mode="constant"):
        return np.pad(x, pad_width, mode=mode)

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def var(self, x, axis=None, keepdims=False):
        return np.var(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def maximum(self, x, y):
        return np.maximum(x, y)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def swapaxes(self, x, axis1, axis2):
        return np.swapaxes(x, axis1, axis2)
