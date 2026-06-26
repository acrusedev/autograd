from __future__ import annotations
import pathlib
import struct
import numpy as npy
from math import prod
from typing import Iterable, List, Optional, Union
from autograd_core import numpy as np

from autograd.helpers import all_values_same, check_shape_compatibility, fetch, fully_flatten, calc_strides, all_int, argfix
from autograd.dtypes import DType, dtypes, to_dtype, dtype_default_float, dtype_default_int
from autograd.dtypes import _from_np_dtypes
from autograd.ops.uop import UOp
from autograd.ops import Ops
from autograd.device import Device
from autograd.scheduler import Scheduler
from autograd.engine.realize import run_schedule
from autograd.mixin.movement import MovementMixin
from autograd.mixin.elementwise import ElementwiseMixin

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
  if isinstance(data, bytes):
    buf_uop = UOp(Ops.BUFFER,dtype,src=(),arg=(data,shape,strides)) # src is empty we pass everything as args
    return buf_uop
  raw_bytes = struct.pack(fmt, *data if hasattr(data,'__len__') else data)
  buf_uop = UOp(Ops.BUFFER,dtype,src=(),arg=(raw_bytes,shape,strides)) # src is empty we pass everything as args
  return buf_uop

def _normalize_shape(s: Optional[Iterable]) -> Optional[tuple[int, ...]]:
  # since shape can be either of list|tuple we need to normalize it
  if s is None:
    return None
  ret = tuple(s)
  assert all_int(ret), "shape should contain ints only"
  return ret

class Tensor(MovementMixin, ElementwiseMixin):
  def __init__(
      self,
      data: Union[UOp, pathlib.Path, List, bytes, memoryview, npy.ndarray, None],
      shape: Optional[Iterable] = None,
      dtype: Optional[DType|str] = None,
      offset: int = 0,
      device:str|None=None,
      realized:bool|None=None,
    ):
    _dtype: DType|None = to_dtype(dtype) if dtype and isinstance(dtype, str) else dtype
    _shape = _normalize_shape(shape)
    self._offset = offset
    self._buffer = None
    # offset is required to implement __getitem__
    self._device = Device

    if isinstance(data, UOp):
      assert _dtype is None or _dtype == data.dtype, "datatype mismatch"
      if _shape is not None: assert _shape == data.shape, "shape mismatch"
      self.uop = data
      self._dtype = data.dtype
      self._shape = _shape if _shape is not None else data.shape
      self._strides = data.strides
      self._offset = data.offset
    elif isinstance(data, (list, tuple)):
      flat = fully_flatten(data)
      if _dtype is None:
        if flat and all(isinstance(el, bool) for el in flat):
          _dtype = dtypes.boolean
        else:
          _dtype = dtype_default_int if all_int(flat) else dtype_default_float
      inferred_shape = get_shape(data)
      self._shape = _shape if _shape is not None else inferred_shape
      if not check_shape_compatibility(inferred_shape, self._shape):
        raise ValueError(f"shape {self._shape} is incompatible with data shape {inferred_shape}")
      self._dtype = _dtype
      self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
      self.uop = _uop_from_data(flat, self._dtype, self._shape, self._strides)
    elif isinstance(data, bytes):
      if _dtype is None:
        raise ValueError("cannot guess datatype from bytes")
      self._dtype = _dtype
      itemsize = self.dtype.bitsize // 8
      if len(data) % itemsize != 0:
        raise ValueError(f"buffer length {len(data)} is not divisible by dtype itemsize {itemsize}")
      self._shape = _shape if _shape is not None else (len(data) // itemsize,)
      if prod(self._shape) * itemsize != len(data):
        raise ValueError(f"shape {self._shape} is incompatible with buffer length {len(data)} and dtype {self.dtype.name}")
      self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
      self.uop = _uop_from_data(data, self.dtype, self._shape, self._strides)
    elif isinstance(data, npy.ndarray):
      self._dtype = _dtype or _from_np_dtypes(data.dtype)
      if data.shape == ():
        self._shape = ()
        self._strides = ()
        self.uop = UOp(op=Ops.CONST, dtype=self.dtype, arg=(data.item(),))
      else:
        if not data.flags.c_contiguous:
          data = npy.ascontiguousarray(data)
        inferred_shape = tuple(data.shape)
        self._shape = _shape if _shape is not None else inferred_shape
        if prod(self._shape) != prod(inferred_shape):
          raise ValueError(f"shape {self._shape} is incompatible with ndarray shape {inferred_shape}")
        self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
        self.uop = _uop_from_data(data.tobytes(), dtype=self.dtype, shape=self._shape, strides=self._strides)
    else:
      raise TypeError(f"unsupported data type: {type(data)!r}")

  @staticmethod
  def from_url(url: str, **kwargs) -> Tensor:
    return Tensor(fetch(url=url), **kwargs)

  def __repr__(self):
    if not self._buffer:
      return f"Tensor <shape={self.shape}, strides={self.strides}>, dtype={self.dtype.name}, op={self.uop}>"
    else:
      return np(self._buffer)

  @property
  def shape(self) -> tuple[int,...]:
    return self._shape

  @property
  def strides(self) -> tuple[int,...]:
    return self._strides

  @property
  def dtype(self) -> DType:
    return self._dtype

  @property
  def offset(self) -> int:
    return self.uop.offset

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

  @staticmethod
  def frombuffer(data: bytes, **kwargs):
    return Tensor(data, **kwargs)

  @classmethod
  def from_np(cls,arr: npy.ndarray) -> Tensor:
    return Tensor(arr)

def expand(a: Tensor, b: Tensor, target_shape: tuple[int,...]) -> tuple[Tensor, Tensor]:
  a = a.expand(target_shape)
  b = b.expand(target_shape)
  return a,b
