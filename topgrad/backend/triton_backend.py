import numpy as np
import torch

from .base import Backend


class TritonBackend(Backend):
    name = "triton"

    def __init__(self):
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Triton backend requires Triton to be installed"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Triton backend requires CUDA to be available")
        super().__init__()
        self.xp = np
        self.device = torch.device("cuda")

    def _normalize_dtype(self, dtype):
        if dtype is None:
            return torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        np_dtype = np.dtype(dtype)
        mapping = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.bool_): torch.bool,
        }
        return mapping.get(np_dtype, torch.float32)

    def is_array(self, x):
        return isinstance(x, torch.Tensor)

    def asarray(self, data, dtype=None):
        dtype = self._normalize_dtype(dtype)
        if isinstance(data, torch.Tensor):
            if data.device != self.device or data.dtype != dtype:
                return data.to(device=self.device, dtype=dtype)
            return data
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        dtype = self._normalize_dtype(dtype)
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        dtype = self._normalize_dtype(dtype)
        return torch.ones(shape, dtype=dtype, device=self.device)

    def ones_like(self, x):
        return torch.ones_like(x)

    def full(self, shape, fill_value, dtype=None):
        dtype = self._normalize_dtype(dtype or getattr(fill_value, "dtype", None))
        return torch.full(shape, fill_value, dtype=dtype, device=self.device)

    def randn(self, shape):
        return torch.randn(*shape, dtype=torch.float32, device=self.device)

    def empty(self, shape, dtype=None):
        dtype = self._normalize_dtype(dtype)
        return torch.empty(shape, dtype=dtype, device=self.device)

    def pad(self, x, pad_width, mode="constant"):
        if mode != "constant":
            raise NotImplementedError("only constant padding is supported")
        pad_width = tuple((int(a), int(b)) for a, b in pad_width)
        out_shape = tuple(
            int(dim) + before + after
            for dim, (before, after) in zip(x.shape, pad_width)
        )
        out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        slices = tuple(
            slice(before, before + int(dim))
            for dim, (before, after) in zip(x.shape, pad_width)
        )
        out[slices] = x
        return out

    def sum(self, x, axis=None, keepdims=False):
        return torch.sum(x, dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return torch.mean(x, dim=axis, keepdim=keepdims)

    def var(self, x, axis=None, keepdims=False):
        return torch.var(x, dim=axis, keepdim=keepdims, unbiased=False)

    def max(self, x, axis=None, keepdims=False):
        if axis is None:
            return torch.max(x)
        return torch.amax(x, dim=axis, keepdim=keepdims)

    def exp(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return torch.exp(x)

    def log(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return torch.log(x)

    def sqrt(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return torch.sqrt(x)

    def maximum(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self.device)
        if isinstance(y, torch.Tensor):
            if y.device != x.device or y.dtype != x.dtype:
                y = y.to(device=x.device, dtype=x.dtype)
        else:
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        return torch.maximum(x, y)

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def swapaxes(self, x, axis1, axis2):
        return torch.swapaxes(x, axis1, axis2)
