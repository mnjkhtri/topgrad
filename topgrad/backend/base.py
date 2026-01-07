from typing import Any, Iterable, Union


class Backend:
    """Backend interface: all ops required by the autograd ops live here."""

    name = "base"

    def __init__(self):
        self.xp = None  # array module, e.g., numpy

    @property
    def float32(self):
        return self.xp.float32

    def is_array(self, x: Any) -> bool:
        # Used by tensor construction to detect backend-native arrays.
        raise NotImplementedError

    def asarray(self, data: Any, dtype=None):
        # Used by tensor construction to coerce Python data into backend arrays.
        raise NotImplementedError

    def zeros(self, shape: Union[int, Iterable[int]], dtype=None):
        # Used by Tensor.zeros and Conv._col2im buffer allocation.
        raise NotImplementedError

    def ones(self, shape: Union[int, Iterable[int]], dtype=None):
        # Used by Tensor.ones for parameter initialization.
        raise NotImplementedError

    def ones_like(self, x):
        # Used by Tensor.backward to seed the initial gradient.
        raise NotImplementedError

    def full(self, shape: Union[int, Iterable[int]], fill_value, dtype=None):
        # Used by Sum/Mean backward in topgrad/ops/op01_reductions.py.
        raise NotImplementedError

    def randn(self, shape: Union[int, Iterable[int]]):
        # Used by Tensor.randn for random initialization.
        raise NotImplementedError

    def empty(self, shape: Union[int, Iterable[int]], dtype=None):
        # Used by Conv._im2col to build an unfold buffer.
        raise NotImplementedError

    def pad(self, x, pad_width, mode="constant"):
        # Used by Conv._im2col to pad NHWC inputs.
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False):
        # Used by Sum/Mean ops and reductions in Linear/Conv/Norms/Attention.
        raise NotImplementedError

    def mean(self, x, axis=None, keepdims=False):
        # Used by Mean, BatchNorm, and LayerNorm.
        raise NotImplementedError

    def var(self, x, axis=None, keepdims=False):
        # Used by BatchNorm and LayerNorm.
        raise NotImplementedError

    def max(self, x, axis=None, keepdims=False):
        # Used by LogSoftmax and Attention softmax stabilization.
        raise NotImplementedError

    def exp(self, x):
        # Used by LogSoftmax and Attention softmax.
        raise NotImplementedError

    def log(self, x):
        # Used by LogSoftmax.
        raise NotImplementedError

    def sqrt(self, x):
        # Used by BatchNorm, LayerNorm, and Attention scaling.
        raise NotImplementedError

    def maximum(self, x, y):
        # Used by ReLU.
        raise NotImplementedError

    def matmul(self, a, b):
        # Used by Linear, Conv (GEMM), and Attention.
        raise NotImplementedError

    def swapaxes(self, x, axis1, axis2):
        # Used by Attention to form QK^T and for backprop.
        raise NotImplementedError

    def numel(self, shape: Iterable[int]) -> int:
        n = 1
        for d in shape:
            n *= int(d)
        return n
