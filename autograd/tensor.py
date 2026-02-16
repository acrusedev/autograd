from typing import Iterable, List, Optional, Union, Tuple
from math import prod
import pathlib
import struct
from operator import countOf

from autograd.helpers import all_values_same, check_shape_compatibility, fetch, fully_flatten, calc_strides, all_int
from autograd.dtypes import DType, dtypes, to_dtype

from autograd_core import Buffer

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
  def __init__(self, data: Union[pathlib.Path, List, bytes, memoryview, None], shape: Optional[Iterable] = None, dtype: Optional[DType] = None):
    _dtype: DType|None = to_dtype(dtype) if dtype is not None else None
    del dtype # from now on we should only use _dtype which has been 'validated'
    if isinstance(data, pathlib.Path):
      raw = data.read_bytes()
      dtype = _dtype or dtypes.uint8 # trust user or read bytes
      self.shape = tuple(shape) if shape else (len(raw),)
      self.dtype = dtype
      self.strides = calc_strides(self.shape, self.dtype.bitsize // 8) # the byte offset in memory to step on each dimension then traversing an array
      self._buffer = Buffer(raw, self.shape, self.strides, self.dtype.fmt)
    if isinstance(data, (bytes, memoryview)):
      raw = data if isinstance(data, bytes) else data.tobytes()
      self.dtype = _dtype or dtypes.uint8
      self.shape = tuple(shape) if shape else (len(raw) // (self.dtype.bitsize // 8),)
      if dtype is None: raise ValueError("dtype is required when data is bytes")
      self.strides = calc_strides(self.shape, self.dtype.bitsize // 8)
      self._buffer = Buffer(raw, self.shape, self.strides, self.dtype.fmt)
    if isinstance(data, (List, Tuple)):
      if _dtype is None:
        if (d := fully_flatten(data)) and all(isinstance(el, bool) for el in d): _dtype = dtypes.boolean
        else: _dtype = dtypes.dtype_default_int if all_int(d) else dtypes.dtype_default_float
      self.dtype = _dtype
      self.shape = get_shape(data)
      self.strides = calc_strides(self.shape, self.dtype.bitsize // 8)
      fmt = f"{len(data)}{self.dtype.fmt}" if self.dtype is not None else ""
      raw = struct.pack(fmt, *data)
      self._buffer = Buffer(raw, self.shape, self.strides, self.dtype.fmt)

  @property
  def data(self) -> memoryview:
    return memoryview(self._buffer) # type: ignore

  @staticmethod
  def from_url(url: str, **kwargs) -> 'Tensor':
    return Tensor(fetch(url=url), **kwargs)

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

  def __repr__(self):
    return f"Tensor({self.data.tolist()}, dtype={self.dtype})"

  def __getitem__(self, x) -> 'Tensor':
    return self._buffer[x]

  @staticmethod
  def zeros(*shape, **kwargs) -> 'Tensor':
    """
    Create a tensor with the given shape filled with int32 0s, you can cast it later as any type
    """
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
      shape = tuple(shape[0])
    else:
      shape = tuple(shape)
    if not shape:
      raise TypeError("zeros() missing shape")
    if not all(isinstance(x, int) and x >= 0 for x in shape):
      raise ValueError("shape must be non-negative integers")
    data = [0] * prod(shape)
    return Tensor(data, shape, **kwargs)
