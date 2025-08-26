import numpy as np
from topgrad.tensor import Tensor
from topgrad.optim import SGD
from .utils import mnist

X_train, Y_train, X_valid, Y_valid = mnist()
X_train = X_train.reshape(-1, 784) / 255.0
X_valid = X_valid.reshape(-1, 784) / 255.0

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
    
BATCH_SIZE = 256
model = TopMLPNet()
optim = SGD([model.l1, model.b1, model.l2, model.b2], lr=0.001)

i = 0
try:
    print("Starting training... Press Ctrl+C to stop.")
    while True:
        samp = np.random.randint(0, X_train.shape[0], size=(BATCH_SIZE))
        x = Tensor(X_train[samp])
        y_labels = Y_train[samp]
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
        if i % 100 == 0:
            print(f"step {i:5d} | loss: {loss.data.item():.4f} | minbatch accuracy: {accuracy:.4f}")
        i += 1
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user. Running final evaluation...")

log_probs = model.forward(Tensor(X_valid)).logsoftmax()
y_preds = np.argmax(log_probs.data, axis=1) # no need to find actually probablity just do argmax
eval_accuracy = (Y_valid == y_preds).mean()
print("final test set accuracy is", eval_accuracy)