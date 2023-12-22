import numpy as np
from tqdm import tqdm
import requests, gzip, os, hashlib

def fetch_mnist(url='http://yann.lecun.com/exdb/mnist/'):
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    
    if os.path.isfile(fp):
        with open(fp, "rb") as f: dat = f.read()
    else:
        with open(fp, "wb") as f: dat = requests.get(url).content; f.write(dat)

    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
  
url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

X_train = fetch_mnist(url+files[0])[0x10:].reshape((-1, 28, 28))
Y_train = fetch_mnist(url+files[1])[8:]
X_valid = fetch_mnist(url+files[2])[0x10:].reshape((-1, 28, 28))
Y_valid = fetch_mnist(url+files[3])[8:]
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

#----

from topgrad.tensor import Tensor
from topgrad.optim import SGD

def layer_init_uniform(m, h):
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
    return ret.astype(np.float32)

class TopNet:
    def __init__(self):
        self.l1 = Tensor(layer_init_uniform(784, 128))
        self.l2 = Tensor(layer_init_uniform(128, 10))

    def forward(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()
    
model = TopNet()
optim = SGD([model.l1, model.l2], lr=0.001)

BS = 128
losses, accuracies = [], []

for t in tqdm(range(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp].reshape((-1, 28*28)))
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)

    y[range(y.shape[0]), Y] = -10.0

    y = Tensor(y)

    out = model.forward(x)

    loss = out.mul(y).mean()
    loss.backward()
    optim.step()

    cat = np.argmax(out.data, axis=1)
    accuracy = (cat==Y).mean()

    loss = loss.data
    losses.append(loss)
    accuracies.append(accuracy)

def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_valid.reshape((-1, 28*28))))
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_valid == Y_test_preds).mean()

accuracy = numpy_eval()
print("Test set accuracy is", accuracy)
assert accuracy > 0.95