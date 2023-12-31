import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super(SGD, self).__init__(params)
        self.lr = lr

    def step(self):
        for t in self.params:
            t.data -= self.lr * t.grad