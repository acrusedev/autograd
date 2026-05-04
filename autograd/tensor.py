from __future__ import annotations
from typing import Iterable, List, Optional, Union
import pathlib
import struct
from autograd_core import numpy as np

from autograd.helpers import all_values_same, check_shape_compatibility, fetch, fully_flatten, calc_strides, all_int
from autograd.dtypes import DType, dtypes, to_dtype, dtype_default_float, dtype_default_int, least_common_dtype, as_dtype
from autograd.ops.uop import UOp
from autograd.ops import Ops
from autograd.device import Device
from autograd.scheduler import Scheduler
from autograd.engine.realize import run_schedule
from autograd.mixin.movement import MovementMixin

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

def _uop_from_data(data: list|tuple|bytes, dtype: DType, shape, strides) -> UOp:
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
  assert all_int(ret), "shape should contain ints only"
  return ret

class Tensor(MovementMixin):
  def __init__(self, data: Union[UOp, pathlib.Path, List, bytes, memoryview, None], shape: Optional[Iterable] = None, dtype: Optional[DType] = None, requires_grad:Optional[bool]=False, device:str|None=None, realized:bool|None=None):
    _dtype: DType|None = to_dtype(dtype) if dtype is not None else None
    _shape = _normalize_shape(shape)
    self._buffer = None

    """
    not every Tensor will require backprop, every Tensor that will be created 
    as a result of a UOp done on a requires_grad Tensor, will also share the 
    requires_grad value
    """
    self.requires_grad = requires_grad
    self._device = Device

    if isinstance(data, UOp):
      assert _dtype is None or _dtype == data.dtype, "datatype mismatch"
      if _shape: assert _shape == data.shape, "shape mismatch"
      self.uop = data
      self.dtype = data.dtype
      if data.op == Ops.BUFFER:
        self._shape = data.arg[1]
      elif _shape is not None:
        self._shape = _shape
      else:
        self._shape = data._shape
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
      self.uop = _uop_from_data(flat, self.dtype, self._shape, self._strides)
    else:
      raise TypeError(f"unsupported data type: {type(data)!r}")
    self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)

  @staticmethod
  def from_url(url: str, **kwargs) -> Tensor:
    return Tensor(fetch(url=url), **kwargs)

  def __repr__(self):
    return f"Tensor <shape={self.shape}, strides={self.strides}>, dtype={self.dtype.name}>"

  @property
  def shape(self) -> tuple[int,...]:
    return self.uop.shape

  @property
  def strides(self) -> tuple[int,...]:
    return self.uop.strides

  def _make_schedule(self):
    return Scheduler(self.uop).nodes

  def realize(self) -> Tensor:
    # actually compute the graph
    if not self._buffer:
      self._buffer = run_schedule(self._make_schedule())
    return self

  def numpy(self):
    if not self._buffer:
      self.realize()
    print(np(self._buffer))

  def __add__(self, other: Tensor) -> Tensor: # todo: later scheduler should allow adding ints and floats by broadcasting
    assert isinstance(other, (Tensor, int, float)), "can add only a tensor or int or float to a tensor"
    assert self.shape == other.shape, "at this moment broadcasting is not supported, cannot add tensors with different shapes"
    if isinstance(other,Tensor):
      return Tensor(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, other.uop)))
    return Tensor(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, UOp(Ops.CONST, dtype=as_dtype(other),arg=(other,)))))
