from time import monotonic
from autograd.tensor import Tensor
from autograd.helpers import fetch

def mnist():
  base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
  # print(fetch(base_url + "train-images-idx3-ubyte.gz").read_bytes())
  # def _mnist(file): return Tensor.frombuffer(fetch(base_url + file).read_bytes(), dtype='uint8')
  return Tensor.frombuffer(fetch(base_url+"train-images-idx3-ubyte.gz").read_bytes(), dtype='uint8')
  # return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("train-labels-idx1-ubyte.gz")[8:], _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("t10k-labels-idx1-ubyte.gz")[8:]

if __name__=="__main__":
  # a = mnist()
  a = [1,2,3,4,5,6,7,8,9]
  a = a[1:4]
  print(a)
  b = [
        [
          [1,2,3],
          [1,2,3],
          [1,2,3],
        ],
        [
          [1,2,3],
          [1,2,3],
          [1,2,3],
        ],
        [
          [1,2,3],
          [1,2,3],
          [1,2,3],
        ],
        [
          [1,2,3],
          [1,2,3],
          [1,2,3],
        ]
      ]
  ba = b[1::2]
  bt = Tensor(b)
  bat = Tensor(ba)
  print(f"shape={bt.shape}")
  print(f"shape={bat.shape}")
  print(b)
  print(ba)
  c = [1]
  print(Tensor(c)[1,2:3:2])
