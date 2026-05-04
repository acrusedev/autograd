from time import monotonic
from autograd.tensor import Tensor

if __name__=="__main__":
  a = Tensor([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])
  b = Tensor([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9])
  e = a + b
  start = monotonic()
  e.realize()
  end= monotonic()
  print(f"{end-start}")
  start = monotonic()
  e.realize()
  end= monotonic()
  print(f"{end-start}")
  e.numpy()
