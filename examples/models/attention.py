from topgrad.tensor import Tensor


class TopAttentionNet:
    def __init__(self):
        # (B, 28, 28)
        self.wq1, self.wk1, self.wv1 = (
            Tensor.He(28, 28),
            Tensor.He(28, 28),
            Tensor.He(28, 28),
        )
        self.lg1, self.lb1 = Tensor.ones(28), Tensor.zeros(28)
        # (B, 28, 28)
        self.fw2, self.fb2 = Tensor.He(28, 64), Tensor.zeros(64)
        self.fw3, self.fb3 = Tensor.He(64, 28), Tensor.zeros(28)
        # (B, 28, 28)
        self.l4, self.b4 = Tensor.He(28 * 28, 10), Tensor.zeros(10)
        # (B, 10)

    def forward(self, x):
        # (B, 28, 28)
        x = (
            x.attention(self.wq1, self.wk1, self.wv1)
            .layernorm(self.lg1, self.lb1)
            .add(x)
        )  # are you pre?
        # x = x.attention(self.wq1, self.wk1, self.wv1).add(x).layernorm(self.lg1, self.lb1) # or pro? i am pro, resnet was pre
        batch, rows, cols = x.shape
        x_2d = x.reshape((batch * rows, cols))
        x_2d = x_2d.linear(self.fw2, self.fb2).relu().linear(self.fw3, self.fb3)
        x = x_2d.reshape((batch, rows, cols)).add(x)  # is relu should be?
        # (B, 28, 28)
        x = x.reshape((x.shape[0], 784))
        x = x.linear(self.l4, self.b4)
        # (B, 10)
        return x

    def parameters(self):
        return [
            self.wq1,
            self.wk1,
            self.wv1,
            self.lg1,
            self.lb1,
            self.fw2,
            self.fb2,
            self.fw3,
            self.fb3,
            self.l4,
            self.b4,
        ]
