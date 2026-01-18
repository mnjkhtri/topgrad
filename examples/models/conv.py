from topgrad.tensor import Tensor


class TopConvNet:
    def __init__(self):
        # (B, 28, 28, 1)
        self.k1, self.b1 = Tensor.He(3, 3, 1, 8), Tensor.zeros(8)
        # (B, 14, 14, 8)
        self.k2, self.b2 = Tensor.He(3, 3, 8, 16), Tensor.zeros(16)
        # (B, 7, 7, 16)
        # (B, 784)
        self.l3, self.b3 = Tensor.He(784, 10), Tensor.zeros(10)
        # (B, 10)

    def forward(self, x):
        # (B, 28, 28, 1)
        x = x.conv(self.k1, self.b1, (2, 2), (1, 1)).relu()
        # (B, 14, 14, 8)
        x = x.conv(self.k2, self.b2, (2, 2), (1, 1)).relu()
        # (B, 7, 7, 16)
        x = x.reshape((x.shape[0], 784))
        # (B, 784)
        x = x.linear(self.l3, self.b3)
        # (B, 10)
        return x

    def parameters(self):
        return [self.k1, self.b1, self.k2, self.b2, self.l3, self.b3]

