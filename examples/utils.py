import numpy as np
import requests, gzip, os, hashlib

class MNIST:

    @staticmethod
    def _fetch_mnist_file(url):
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        
        if os.path.isfile(fp):
            with open(fp, "rb") as f: dat = f.read()
        else:
            with open(fp, "wb") as f: dat = requests.get(url).content; f.write(dat)

        return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

    @staticmethod
    def _labels_to_onehot(labels, dtype=np.float32):
        labels = np.asarray(labels, dtype=np.int64)
        probs = np.zeros((labels.shape[0], 10), dtype=dtype)
        probs[np.arange(labels.shape[0]), labels] = 1.0
        return probs

    def load_data(train=True, as_format="images", dtype=np.float32):

        url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

        if train:
            X_raw = MNIST._fetch_mnist_file(url + files[0])[0x10:] / 255.0
            Y_raw = MNIST._fetch_mnist_file(url + files[1])[8:]
        else:
            X_raw = MNIST._fetch_mnist_file(url + files[2])[0x10:] / 255.0
            Y_raw = MNIST._fetch_mnist_file(url + files[3])[8:]

        if as_format == "flat":
            X_raw = X_raw.reshape((-1, 28 * 28))
        elif as_format == "image":
            X_raw = X_raw.reshape((-1, 28, 28))
        elif as_format == "channel":
            X_raw = X_raw.reshape((-1, 28, 28, 1))
        else:
            raise ValueError(f"Unsupported as_format: {as_format}")

        X = X_raw.astype(dtype, copy=False)
        Y = MNIST._labels_to_onehot(Y_raw, dtype=dtype)
        return X, Y
