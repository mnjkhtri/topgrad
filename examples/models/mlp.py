from topgrad.tensor import Tensor


class TopMLPNet:
    def __init__(self):
        self.l1, self.b1 = Tensor.He(784, 128), Tensor.zeros(128)
        self.l2, self.b2 = Tensor.He(128, 10), Tensor.zeros(10)

    def forward(self, x):
        # (B, 784)
        x = x.linear(self.l1, self.b1).relu()
        # (B, 128)
        x = x.linear(self.l2, self.b2)
        # (B, 10)
        return x

    def parameters(self):
        return [self.l1, self.b1, self.l2, self.b2]

