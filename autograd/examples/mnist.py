from time import monotonic
import numpy as np
from autograd.tensor import Tensor

if __name__=="__main__":
  a = Tensor([1,2])
  b = Tensor([2,3])
  start_autograd = monotonic()
  e = a + b
  e.realize()
  end_autograd = monotonic()
  print(e)
  a = np.array([1,2])
  b = np.array([2,3])
  start_numpy = monotonic()
  e = a+b
  end_numpy = monotonic()
  print(e)
  print(f"time autograd {end_autograd-start_autograd}, time numpy {end_numpy-start_numpy}")
