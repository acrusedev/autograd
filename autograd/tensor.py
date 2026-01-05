from typing import Iterable, List, Optional, Union
import pathlib

from autograd.helpers import all_values_same, fetch

def get_shape(x) -> tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str, therefore we need to check both getitem and str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return (0,) # x is a scalar value so its a 0D tensor -> shape(0,)
  """
  NOTE: at this point we know that x is an any dimensional iterable, we need to check if all sub-elements have the same shape
  allowed: [[1,2], [3,4]], [[1,2], [3,5]]
  not allowed: [1, [1,2], [1,2,3]]
  """
  if not all_values_same(element_shape:=[get_shape(sub_list) for sub_list in x]): raise ValueError(f"inhomogeneous shape from {x}")
  return (len(element_shape),) + (element_shape[0] if element_shape else ())

class Tensor:
  def __init__(self, data: Union[pathlib.Path, List], shape: Optional[Iterable] = None):
    if isinstance(data, pathlib.Path):
      with open(data, 'rb') as file:
        stream = file.read()
      self.data = list(stream)
    self.shape = shape if shape else get_shape(self.data) 

  def reshape(self, shape: Iterable):
    self.shape = shape

  @staticmethod
  def from_url(url: str) -> 'Tensor':
    return Tensor(fetch(url=url))

  def __repr__(self):
    return f"Tensor with shape {self.shape}"

