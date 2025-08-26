import numpy as np
from topgrad.tensor import Tensor
from topgrad.optim import SGD
from .utils import mnist

X_train, Y_train, X_valid, Y_valid = mnist()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_valid = X_valid.reshape(-1, 28, 28, 1) / 255.0

class TopConvNet:
    def __init__(self):
        # (B, 28, 28, 1)
        self.k1, self.b1 = Tensor.He(3, 3, 1, 8),  Tensor.zeros(8)
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
    
BATCH_SIZE = 256
model = TopConvNet()
optim = SGD([model.k1, model.b1, model.k2, model.b2, model.l3, model.b3], lr=0.001)

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
        if i % 100 == 0:
            print(f"step {i:5d} | loss: {loss.data.item():.4f} | minbatch accuracy: {accuracy:.4f}")
        i += 1
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user. Running final evaluation...")

log_probs = model.forward(Tensor(X_valid)).logsoftmax()
y_preds = np.argmax(log_probs.data, axis=1) # no need to find actually probablity just do argmax
eval_accuracy = (Y_valid == y_preds).mean()
print("final test set accuracy is", eval_accuracy)