from types import SimpleNamespace as sn

import numpy as np

from topgrad.backend import get_backend
from topgrad.tensor import Tensor


class TopDeepConvNet:
    def __init__(self):
        backend = get_backend()

        def bn_state(channels):
            return sn(
                running_mean=backend.wrap(np.zeros(channels, dtype=np.float32)),
                running_var=backend.wrap(np.ones(channels, dtype=np.float32)),
            )

        # (B, 28, 28, 1)
        self.k1, self.b1 = Tensor.He(3, 3, 1, 32), Tensor.zeros(32)
        self.bg1, self.bb1, self.st1 = (
            Tensor.ones(32),
            Tensor.zeros(32),
            bn_state(32),
        )
        # (B, 28, 28, 32)
        self.k2, self.b2 = Tensor.He(3, 3, 32, 32), Tensor.zeros(32)
        self.bg2, self.bb2, self.st2 = (
            Tensor.ones(32),
            Tensor.zeros(32),
            bn_state(32),
        )
        # (B, 28, 28, 32)
        self.k3, self.b3 = Tensor.He(3, 3, 32, 64), Tensor.zeros(64)
        self.bg3, self.bb3, self.st3 = (
            Tensor.ones(64),
            Tensor.zeros(64),
            bn_state(64),
        )
        # (B, 14, 14, 64)
        # (B, 12544)
        self.l4, self.b4 = Tensor.He(12544, 1024), Tensor.zeros(1024)
        # (B, 1024)
        self.l5, self.b5 = Tensor.He(1024, 256), Tensor.zeros(256)
        # (B, 256)
        self.l6, self.b6 = Tensor.He(256, 10), Tensor.zeros(10)
        # (B, 10)

    def forward(self, x):
        # (B, 28, 28, 1)
        x = (
            x.conv(self.k1, self.b1, (1, 1), (1, 1))
            .batchnorm(self.bg1, self.bb1, self.st1)
            .relu()
        )
        # (B, 28, 28, 32)
        x = (
            x.conv(self.k2, self.b2, (1, 1), (1, 1))
            .batchnorm(self.bg2, self.bb2, self.st2)
            .add(x)
            .relu()
        )
        # (B, 28, 28, 32)
        x = (
            x.conv(self.k3, self.b3, (2, 2), (1, 1))
            .batchnorm(self.bg3, self.bb3, self.st3)
            .relu()
        )
        # (B, 14, 14, 64)
        x = x.reshape((x.shape[0], 12544))
        # (B, 12544)
        x = x.linear(self.l4, self.b4).relu()
        # (B, 1024)
        x = x.linear(self.l5, self.b5).relu()
        # (B, 256)
        x = x.linear(self.l6, self.b6)
        # (B, 10)
        return x

    def parameters(self):
        return [
            self.k1,
            self.b1,
            self.bg1,
            self.bb1,
            self.k2,
            self.b2,
            self.bg2,
            self.bb2,
            self.k3,
            self.b3,
            self.bg3,
            self.bb3,
            self.l4,
            self.b4,
            self.l5,
            self.b5,
            self.l6,
            self.b6,
        ]
