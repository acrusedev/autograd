from typing import Iterable, List, Optional, Union
from math import prod
import pathlib
import struct

from autograd.helpers import all_values_same, check_shape_compatibility, fetch, fully_flatten, calc_strides, all_int
from autograd.dtypes import DType, dtypes, to_dtype, dtype_default_float, dtype_default_int
from autograd.ops.uop import UOp
from autograd.ops import Ops


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

def _frompy(data: list|tuple|bytes, dtype: DType, shape, strides) -> UOp:
  # get type and flatten data
  fmt = f"{len(data)}{dtype.fmt}"
  raw_bytes = struct.pack(fmt, *data if isinstance(data, (list,tuple)) else data)
  buf_uop = UOp(Ops.BUFFER,dtype,src=(),arg=(raw_bytes,shape,strides)) # src is empty we pass everything as args
  return buf_uop

def _normalize_shape(s: Optional[Iterable]) -> Optional[tuple[int, ...]]:
  # since shape can be either of list|tuple we need to normalize it
  if s is None:
    return None
  ret = tuple(s)
  assert all_int(s), "shape need to contain ints only"
  return ret

class Tensor:
  def __init__(self, data: Union[UOp, pathlib.Path, List, bytes, memoryview, None], shape: Optional[Iterable] = None, dtype: Optional[DType] = None, requires_grad:Optional[bool]=False):
    _dtype: DType|None = to_dtype(dtype) if dtype is not None else None
    _shape = _normalize_shape(shape)

    """
    not every Tensor will require backprop, every Tensor that will be created 
    as a result of a UOp done on a requires_grad Tensor will also share the 
    requires_grad value
    """
    self.requires_grad = requires_grad

    if isinstance(data, UOp):
      assert _dtype is None or _dtype == data.dtype, "datatype mismatch"
      self.uop = data
      self.dtype = data.dtype
      if _shape is not None:
        self._shape = _shape
      elif data.op == Ops.BUFFER and isinstance(data.arg, tuple) and len(data.arg) == 2:
        self._shape = (data.arg[1],)
      else:
        self._shape = ()
    elif isinstance(data, (list, tuple)):
      flat = fully_flatten(data)
      if _dtype is None:
        if flat and all(isinstance(el, bool) for el in flat):
          _dtype = dtypes.boolean
        else:
          _dtype = dtype_default_int if all_int(flat) else dtype_default_float
      inferred_shape = get_shape(data)
      self._shape = _shape if _shape is not None else inferred_shape
      if _shape is not None and not check_shape_compatibility(inferred_shape, _shape):
        raise ValueError(f"shape {_shape} is incompatible with data shape {inferred_shape}")
      self.dtype = _dtype
      self._strides = calc_strides(self._shape, self.dtype.bitsize//8)
      self.uop = _frompy(flat, self.dtype, self._shape, self._strides)
    else:
      raise TypeError(f"unsupported data type: {type(data)!r}")
    self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
    if self.uop.op == Ops.BUFFER and isinstance(self.uop.arg, tuple) and len(self.uop.arg) == 2:
      self._buffer = self.uop.arg[0]
    else:
      self._buffer = b""

  @property
  def data(self) -> memoryview:
    return memoryview(self._buffer) # type: ignore

  @staticmethod
  def from_url(url: str, **kwargs) -> 'Tensor':
    return Tensor(fetch(url=url), **kwargs)

  def __getitem__(self, x) -> 'Tensor':
    return self._buffer[x]

  def __repr__(self):
    return f"Tensor <{self.uop.__repr__()}>"

  def reshape(self, target_shape:list|tuple|int, *args) -> 'Tensor':
    if isinstance(target_shape, int):
      if args: target_shape=(target_shape,)+args
      else: target_shape=(target_shape,)
    else: target_shape=tuple(target_shape)
    assert check_shape_compatibility(self._shape, target_shape), f"cannot convert shape {self._shape} to {target_shape}"
    if check_shape_compatibility==self._shape: return self
    return Tensor(UOp(Ops.RESHAPE,dtype=self.dtype, src=(self.uop,), arg=(target_shape,)))

  @property
  def shape(self) -> tuple[int,...]:
    return self.uop.shape
  @property
  def strides(self) -> tuple[int,...]:
    return self.uop.strides
 
  def realize(self):
    # actually compute the graph
    pass

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
