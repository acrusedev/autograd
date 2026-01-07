from autograd.tensor import Tensor

def mnist():
    base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
    def _mnist(file) -> Tensor: return Tensor.from_url(base_url+file)
    return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape((-1,1,28,28)), _mnist("train-labels-idx1-ubyte.gz")[8:], _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,1,28,28)), _mnist("t10k-labels-idx1-ubyte.gz")[8:]
