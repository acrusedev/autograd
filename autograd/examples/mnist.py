import numpy as np
from autograd.tensor import Tensor
from autograd.helpers import fetch


def mnist():
  base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
  def _mnist(file) -> Tensor: return Tensor.frombuffer(fetch(base_url + file).read_bytes(), dtype='uint8')
  return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("train-labels-idx1-ubyte.gz")[8:], _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("t10k-labels-idx1-ubyte.gz")[8:]


class Layer:
  def __init__(self, layer_input_size: int, layer_output_size: int, is_last: bool | None = False):
    self.weights = Tensor(np.random.randn(layer_input_size, layer_output_size).astype(np.float32) * np.sqrt(2.0 / layer_input_size))
    self.biases = Tensor(np.zeros(layer_output_size).astype(np.float32))
    self.input = None
    self.outputs = None
    self.z = None
    self.is_last = is_last
  def forward(self, inputs: Tensor) -> Tensor:
    self.inputs = inputs
    self.outputs = inputs @ self.weights + self.biases

if __name__=="__main__":
  train_images, train_labels, test_images, test_labels = mnist()
  print(train_images, train_images, test_images, test_labels, sep="\n")