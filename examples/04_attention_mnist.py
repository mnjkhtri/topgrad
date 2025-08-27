import numpy as np
from topgrad.tensor import Tensor
from topgrad.optim import SGD
from dataclasses import dataclass
from .utils import mnist

X_train, Y_train, X_valid, Y_valid = mnist()
X_train = X_train.reshape(-1, 28, 28) / 255.0
X_valid = X_valid.reshape(-1, 28, 28) / 255.0


class TopAttentionNet:
    def __init__(self):
        # (B, 28, 28)
        self.wq1, self.wk1, self.wv1 = Tensor.He(28, 28), Tensor.He(28, 28), Tensor.He(28, 28)
        self.lg1, self.lb1 = Tensor.ones(28), Tensor.zeros(28)
        # (B, 28, 28)
        self.fw2, self.fb2 = Tensor.He(28, 64), Tensor.zeros(64)
        self.fw3, self.fb3 = Tensor.He(64, 28), Tensor.zeros(28)
        # (B, 28, 28)
        self.l4, self.b4 = Tensor.He(28*28, 10), Tensor.zeros(10)
        # (B, 10)

    def forward(self, x):
        x = x.attention(self.wq1, self.wk1, self.wv1).layernorm(self.lg1, self.lb1).add(x) # are you pre?
        # x = x.attention(self.wq1, self.wk1, self.wv1).add(x).layernorm(self.lg1, self.lb1) # or pro? i am pro, resnet was pre
        x = x.linear(self.fw2, self.fb2).relu().linear(self.fw3, self.fb3).add(x) # is relu should be?
        x = x.reshape((x.shape[0], 784))
        x = x.linear(self.l4, self.b4)
        return x

BATCH_SIZE = 8
model = TopAttentionNet()
optim = SGD([
    model.wq1, model.wk1, model.wv1, model.lg1, model.lb1,
    model.fw2, model.fb2,
    model.fw3, model.fb3,
    model.l4, model.b4,
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