import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib
from topgrad.tensor import Tensor
from topgrad.optim import SGD

def fetch_mnist(url='http://yann.lecun.com/exdb/mnist/'):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    
    if os.path.isfile(fp):
        with open(fp, "rb") as f: dat = f.read()
    else:
        with open(fp, "wb") as f: dat = requests.get(url).content; f.write(dat)

    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
  
url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

X_train = fetch_mnist(url+files[0])[0x10:].reshape((-1, 28, 28)) / 255.0
Y_train = fetch_mnist(url+files[1])[8:]
X_valid = fetch_mnist(url+files[2])[0x10:].reshape((-1, 28, 28)) / 255.0
Y_valid = fetch_mnist(url+files[3])[8:]

# print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

class TopNet:
    def __init__(self):
        self.l1, self.b1 = Tensor.He(784, 128), Tensor.zeros(128)
        self.l2, self.b2 = Tensor.He(128, 10), Tensor.zeros(10)

    def forward(self, x):
        return x.linear(self.l1, self.b1).relu().linear(self.l2, self.b2)
    
BATCH_SIZE = 256
model = TopNet()
optim = SGD([model.l1, model.l2], lr=0.001)

i = 0
try:
    print("Starting training... Press Ctrl+C to stop.")
    while True:
        samp = np.random.randint(0, X_train.shape[0], size=(BATCH_SIZE))
        x = Tensor(X_train[samp].reshape((-1, 28*28)))
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

log_probs = model.forward(Tensor(X_valid.reshape((-1, 28*28)))).logsoftmax()
y_preds = np.argmax(log_probs.data, axis=1) # no need to find actually probablity just do argmax
eval_accuracy = (Y_valid == y_preds).mean()
print("final test set accuracy is", eval_accuracy)