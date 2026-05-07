from time import monotonic
from autograd.tensor import Tensor

if __name__=="__main__":
  # a = Tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
  b = Tensor([1,2,3])
  a = Tensor([1,2,3]) # [2,4,6]
  e = a + b
  e.realize()
  print(e.dtype)
  e.numpy()