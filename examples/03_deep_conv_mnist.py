import numpy as np
from topgrad.tensor import Tensor
from topgrad.optim import SGD
from dataclasses import dataclass
from .utils import mnist

X_train, Y_train, X_valid, Y_valid = mnist()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_valid = X_valid.reshape(-1, 28, 28, 1) / 255.0

@dataclass
class BNState:
    running_mean:   np.array
    running_var:    np.array

class TopConvNet:
    def __init__(self):
        # (B, 28, 28, 1)
        self.k1, self.b1 = Tensor.He(3, 3, 1, 32), Tensor.zeros(32)
        self.bg1, self.bb1, self.st1 = Tensor.ones(32), Tensor.zeros(32), BNState(np.zeros(32), np.ones(32))
        # (B, 28, 28, 32)
        self.k2, self.b2 = Tensor.He(3, 3, 32, 32), Tensor.zeros(32)
        self.bg2, self.bb2, self.st2 = Tensor.ones(32), Tensor.zeros(32), BNState(np.zeros(32), np.ones(32))
        # (B, 28, 28, 32)
        self.k3, self.b3 = Tensor.He(3, 3, 32, 64), Tensor.zeros(64)
        self.bg3, self.bb3, self.st3 = Tensor.ones(64), Tensor.zeros(64), BNState(np.zeros(64), np.ones(64))
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
        x = x.conv(self.k1, self.b1, (1, 1), (1, 1)).batchnorm(self.bg1, self.bb1, self.st1).relu()
        # (B, 28, 28, 32)
        x = x.conv(self.k2, self.b2, (1, 1), (1, 1)).batchnorm(self.bg2, self.bb2, self.st2).add(x).relu()
        # (B, 28, 28, 32)
        x = x.conv(self.k3, self.b3, (2, 2), (1, 1)).batchnorm(self.bg3, self.bb3, self.st3).relu()
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
    
BATCH_SIZE = 8
model = TopConvNet()
optim = SGD([
    model.k1, model.b1, model.bg1, model.bb1,
    model.k2, model.b2, model.bg2, model.bb2,
    model.k3, model.b3, model.bg3, model.bb3,
    model.l4, model.b4,
    model.l5, model.b5,
    model.l6, model.b6
], lr=0.001)

i = 0
try:
    print("Starting training... Press Ctrl+C to stop.")
    while True:
        samp = np.random.randint(0, X_train.shape[0], size=(BATCH_SIZE))
        x = Tensor(X_train[samp])
        y_labels =  Y_train[samp]
        y_np = np.zeros((BATCH_SIZE, 10), np.float32)
        y_np[range(BATCH_SIZE), y_labels] = -1 # (negation)
        y_probs = Tensor(y_np)
        log_probs = model.forward(x).logsoftmax()
        loss = log_probs.mul(y_probs).sum() # expected (actually just sum) of y_probs * log likelihood
        loss.backward()
        optim.step()
        optim.zero_grad()
        y_preds = np.argmax(log_probs.data, axis=1)
        accuracy = (y_preds == y_labels).mean()
        if i % 10 == 0:
            print(f"step {i:5d} | loss: {loss.data.item():.4f} | minbatch accuracy: {accuracy:.4f}")
        i += 1
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user. Running final evaluation...")

valid_samp = np.random.randint(0, X_valid.shape[0], size=(32))
log_probs = model.forward(Tensor(X_valid[valid_samp])).logsoftmax()
y_preds = np.argmax(log_probs.data, axis=1) # no need to find actually probablity just do argmax
eval_accuracy = (Y_valid[valid_samp] == y_preds).mean()
print("final test set accuracy is", eval_accuracy)