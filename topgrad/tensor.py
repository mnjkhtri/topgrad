import math

import numpy as np

from topgrad.backend import get_backend


class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self._op = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, grad={self.grad is not None})"

    @classmethod
    def zeros(cls, *shape, dtype=None):
        backend = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(
            backend.wrap(np.zeros(shape, dtype=np.float32 if dtype is None else dtype))
        )

    @classmethod
    def ones(cls, *shape, dtype=None):
        backend = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(
            backend.wrap(np.ones(shape, dtype=np.float32 if dtype is None else dtype))
        )

    @classmethod
    def randn(cls, *shape, dtype=None):
        backend = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return cls(backend.wrap(np.random.randn(*shape).astype(dtype or np.float32)))

    @classmethod
    def He(cls, *shape, dist="normal", dtype=None):
        backend = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        fan_in = shape[0]
        if len(shape) > 1:
            fan_in = 1
            for d in shape[:-1]:
                fan_in *= int(d)

        if dist == "normal":
            std_dev = math.sqrt(2.0 / fan_in)
            return cls(
                backend.wrap(
                    np.random.randn(*shape).astype(dtype or np.float32) * std_dev
                )
            )

        raise NotImplementedError("Backend does not support this distribution.")

    @classmethod
    def Xavier(cls, *shape, dist="normal", dtype=None):
        backend = get_backend()
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        fan_in = shape[0]
        fan_out = shape[0]

        if len(shape) > 1:
            prefix_prod = 1
            for d in shape[:-2]:
                prefix_prod *= int(d)
            fan_in = prefix_prod * shape[-2]
            fan_out = prefix_prod * shape[-1]

        if dist == "normal":
            std_dev = math.sqrt(2.0 / (fan_in + fan_out))
            return cls(
                backend.wrap(
                    np.random.randn(*shape).astype(dtype or np.float32) * std_dev
                )
            )

        raise NotImplementedError("Backend does not support this distribution.")

    def backward(self, allow_fill=True):
        if self._op is None:
            return

        if self.grad is None and allow_fill:
            backend = get_backend()
            self.grad = backend.wrap(np.ones(self.shape, dtype=np.float32))

        assert self.grad is not None

        grads = self._op.backward(self.grad)

        for t, g in zip(self._op.parents, grads):
            if t.grad is None:
                t.grad = g
            else:
                t.grad = t.grad + g

            t.backward(False)

    def sum(self):
        backend = get_backend()
        return backend.ops.Sum.apply(self, backend=backend)

    def mean(self):
        backend = get_backend()
        return backend.ops.Mean.apply(self, backend=backend)

    def relu(self):
        backend = get_backend()
        return backend.ops.ReLU.apply(self, backend=backend)

    def logsoftmax(self):
        backend = get_backend()
        return backend.ops.LogSoftmax.apply(self, backend=backend)

    def mul(self, x):
        backend = get_backend()
        return backend.ops.Mul.apply(self, x, backend=backend)

    def add(self, x):
        backend = get_backend()
        return backend.ops.Add.apply(self, x, backend=backend)

    def linear(self, w_t, b_t):
        backend = get_backend()
        return backend.ops.Linear.apply(self, w_t, b_t, backend=backend)

    def reshape(self, ns):
        backend = get_backend()
        return backend.ops.Reshape.apply(self, ns, backend=backend)

    def conv(self, w_t, b_t, stride, padding):
        backend = get_backend()
        return backend.ops.Conv.apply(self, w_t, b_t, stride, padding, backend=backend)

    def batchnorm(self, gamma_t, beta_t, state, eps=1e-5, momentum=0.1, mode="T"):
        backend = get_backend()
        return backend.ops.BatchNorm.apply(
            self, gamma_t, beta_t, state, eps, momentum, mode, backend=backend
        )

    def layernorm(self, gamma_t, beta_t, eps=1e-5):
        backend = get_backend()
        return backend.ops.LayerNorm.apply(self, gamma_t, beta_t, eps, backend=backend)

    def attention(self, wq_t, wk_t, wv_t):
        backend = get_backend()
        return backend.ops.Attention.apply(self, wq_t, wk_t, wv_t, backend=backend)
