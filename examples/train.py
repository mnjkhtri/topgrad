import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace as sn

from topgrad.tensor import Tensor
from topgrad.optim import SGD

from .utils import MNIST

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

    def parameters(self):
        return [self.k1, self.b1, self.k2, self.b2, self.l3, self.b3]

class TopDeepConvNet:
    def __init__(self):
        # (B, 28, 28, 1)
        self.k1, self.b1 = Tensor.He(3, 3, 1, 32), Tensor.zeros(32)
        self.bg1, self.bb1, self.st1 = Tensor.ones(32), Tensor.zeros(32), sn(np.zeros(32), np.ones(32))
        # (B, 28, 28, 32)
        self.k2, self.b2 = Tensor.He(3, 3, 32, 32), Tensor.zeros(32)
        self.bg2, self.bb2, self.st2 = Tensor.ones(32), Tensor.zeros(32), sn(np.zeros(32), np.ones(32))
        # (B, 28, 28, 32)
        self.k3, self.b3 = Tensor.He(3, 3, 32, 64), Tensor.zeros(64)
        self.bg3, self.bb3, self.st3 = Tensor.ones(64), Tensor.zeros(64), sn(np.zeros(64), np.ones(64))
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
    
    def parameters(self):
        return [
            self.k1, self.b1, self.bg1, self.bb1,
            self.k2, self.b2, self.bg2, self.bb2,
            self.k3, self.b3, self.bg3, self.bb3,
            self.l4, self.b4,
            self.l5, self.b5,
            self.l6, self.b6
        ]

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
        # (B, 28, 28)
        x = x.attention(self.wq1, self.wk1, self.wv1).layernorm(self.lg1, self.lb1).add(x) # are you pre?
        # x = x.attention(self.wq1, self.wk1, self.wv1).add(x).layernorm(self.lg1, self.lb1) # or pro? i am pro, resnet was pre
        x = x.linear(self.fw2, self.fb2).relu().linear(self.fw3, self.fb3).add(x) # is relu should be?
        # (B, 28, 28)
        x = x.reshape((x.shape[0], 784))
        x = x.linear(self.l4, self.b4)
        # (B, 10)
        return x

    def parameters(self):    
        return [
            self.wq1, self.wk1, self.wv1, self.lg1, self.lb1,
            self.fw2, self.fb2,
            self.fw3, self.fb3,
            self.l4, self.b4,
        ]

# X, Y = MNIST.load_data(as_format='flat')
# X_valid, Y_valid = MNIST.load_data(train=False, as_format='flat')
# model = TopMLPNet()
# BATCH_SIZE = 256

# X, Y = MNIST.load_data(as_format='image')
# X_valid, Y_valid = MNIST.load_data(train=False, as_format='channel')
# model = TopConvNet()
# BATCH_SIZE = 32

# X, Y = MNIST.load_data(as_format='image')
# X_valid, Y_valid = MNIST.load_data(train=False, as_format='channel')
# model = TopDeepConvNet()
# BATCH_SIZE = 8

X, Y = MNIST.load_data(as_format='image')
X_valid, Y_valid = MNIST.load_data(train=False, as_format='image')
model = TopAttentionNet()
BATCH_SIZE = 256

optim = SGD(model.parameters(), lr=0.0001)

train_losses, valid_losses = [], []
train_accuracies, valid_accuracies = [], []
steps = []

def plot_metrics():
    plt.figure(figsize=(12,5))

    # Loss plot
    plt.subplot(1,2,1)
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, valid_losses, label="valid")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("train/valid loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1,2,2)
    plt.plot(steps, train_accuracies, label="train")
    plt.plot(steps, valid_accuracies, label="valid")
    plt.xlabel("steps")
    plt.ylabel("accuracy")
    plt.title("train/valid accuracy")
    plt.legend()

    plt.show()

def eval(sample_size):

    assert sample_size < X_valid.shape[0]

    samp = np.random.randint(0, X_valid.shape[0], size=(sample_size))
    x, y_probs = Tensor(X_valid[samp]), Tensor(Y_valid[samp])
    log_probs = model.forward(x).logsoftmax()
    loss = log_probs.mul(y_probs).neg().sum()
    # no backward

    y_labels = np.argmax(y_probs.data, axis=1)
    y_pred_labels = np.argmax(log_probs.data, axis=1)

    accuracy = 100 * (y_labels == y_pred_labels).mean()
    return loss, accuracy

def train():

    i = 0
    try:
        print("Starting training... Press Ctrl+C to stop.")
        while True:
            samp = np.random.randint(0, X.shape[0], size=(BATCH_SIZE))
            x, y_probs = Tensor(X[samp]), Tensor(Y[samp])

            log_probs = model.forward(x).logsoftmax()
            loss = log_probs.mul(y_probs).neg().sum()

            loss.backward()
            optim.step()
            optim.zero_grad()


            if i % 1 == 0:

                # TRAIN:
                y_labels = np.argmax(y_probs.data, axis=1)
                y_pred_labels = np.argmax(log_probs.data, axis=1)
                accuracy = 100 * (y_labels == y_pred_labels).mean()

                # VALID:
                eval_loss, eval_accuracy = eval(BATCH_SIZE)

                print(f"minbatch {i:5d} | loss (train/valid): {loss.data:.4f}, {eval_loss.data:.4f} | accuracy(train/valid): {accuracy:.2f}, {eval_accuracy:.2f}")

            steps.append(i)
            train_losses.append(loss.data)
            valid_losses.append(eval_loss.data)
            train_accuracies.append(accuracy)
            valid_accuracies.append(eval_accuracy)

            i += 1

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Thanks for training with us. Here is your plot:")
        plot_metrics()

if __name__ == "__main__":

    train()