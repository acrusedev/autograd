from autograd.tensor import Tensor

if __name__=="__main__":
  a = Tensor([1,2])
  b = Tensor([2,3])
  e = a + b
  e.realize()