import numpy as np
import requests, gzip, os, hashlib

def mnist():

    def fetch_mnist(url):
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        
        if os.path.isfile(fp):
            with open(fp, "rb") as f: dat = f.read()
        else:
            with open(fp, "wb") as f: dat = requests.get(url).content; f.write(dat)

        return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    
    url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    X_train, Y_train = fetch_mnist(url+files[0])[0x10:].reshape((-1, 28, 28)), fetch_mnist(url+files[1])[8:]
    X_valid, Y_valid = fetch_mnist(url+files[2])[0x10:].reshape((-1, 28, 28)), fetch_mnist(url+files[3])[8:]

    return X_train, Y_train, X_valid, Y_valid