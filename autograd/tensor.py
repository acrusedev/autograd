from typing import Iterable, List, Optional, Union
from math import prod
import pathlib
from operator import countOf

from autograd.helpers import all_values_same, check_shape_compatibility, fetch, fully_flatten

def get_shape(x) -> tuple[int, ...]:
  # NOTE: str is special because __getitem__ on a str is still a str, therefore we need to check both getitem and str
  if not hasattr(x, "__len__") or not hasattr(x, "__getitem__") or isinstance(x, str) or (hasattr(x, "shape") and x.shape == ()): return () # x is a scalar value so its a 0D tensor -> shape(0,)
  """
  NOTE: at this point we know that x is an any dimensional iterable, we need to check if all sub-elements have the same shape
  allowed: [[1,2], [3,4], [4,5]]
  not allowed: [1, [1,2], [1,2,3]]
  """
  if not all_values_same(element_shape:=[get_shape(element) for element in x]): raise ValueError(f"inhomogeneous shape from {x}")

  return (len(element_shape),) + (element_shape[0] if element_shape else ())

class Tensor:
  def __init__(self, data: Union[pathlib.Path, List], shape: Optional[Iterable] = None):
    if isinstance(data, pathlib.Path):
      with open(data, 'rb') as file:
        stream = file.read()
      # self.data needs to be flattened to a 1D list
      self.data = fully_flatten(list(stream)) # TODO: later this will be changed by specifying dtype explicitly
    else:
      self.data = data
    self.shape = shape if shape else get_shape(self.data)

  def reshape(self, shape: tuple[int,...]) -> 'Tensor':
    # allow only for positive integers except for -1
    if not all(isinstance(element, int) and element >= -1 for element in shape): raise ValueError("only positive integers or -1 are allowed for shape")
    # check if the new shape is compatible with the current shape
    # allow for guesssing one parameter by using -1
    if countOf(shape, -1) > 1: raise ValueError("only one dimension can be -1")
    s = [s for s in shape]
    if -1 in s:
      s[s.index(-1)] = len(self.data) // (-1 * prod(s))
      shape = tuple(s)
    if not check_shape_compatibility(self.shape, shape): raise ValueError(f"new shape {shape} is not compatible with current shape {self.shape}")
    self.shape = shape
    return self


  @staticmethod
  def from_url(url: str) -> 'Tensor':
    return Tensor(fetch(url=url))

  def __repr__(self):
    return f"Tensor with shape {self.shape}"

  def __getitem__(self, x) -> 'Tensor':
    x = Tensor(self.data[x])
    return x
