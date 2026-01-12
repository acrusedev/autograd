import numpy as np
from autograd.helpers import fetch

def mnist():
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    def _mnist(file): return np.frombuffer(fetch(base_url + file).read_bytes(), np.uint8)
    return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape((-1,1,28,28)), _mnist("train-labels-idx1-ubyte.gz")[8:], _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,1,28,28)), _mnist("t10k-labels-idx1-ubyte.gz")[8:]

train_images, train_labels, test_images, test_labels = mnist()
print(train_images.shape)
