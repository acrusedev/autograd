from dataclasses import dataclass
from typing import Tuple, Any

from autograd.ops import Ops
from autograd.dtypes import DType

@dataclass(frozen=True)
class UOp:
  op: Ops
  dtype: DType # target dtype after operation
  src: Tuple['UOp',...]=tuple()
  arg: Any = None # this depending on the operation will be different data structures

  def __repr__(self):
    return f"UOp <{self.op}, dtype={self.dtype.name}>"

  @staticmethod
  def new_buffer(self):
    pass

  def _shape(self):
    """
    for each node in the computation graph this calculates the shape of a tensor that were to be created at that point
    """
    pass

  @property
  def shape(self):
    return self._shape()
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
    self.strides = calc_strides(shape, self.dtype.bitsize // 8)
    self._buffer.reshape(self.shape, self.strides)
    return self