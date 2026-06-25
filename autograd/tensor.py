from __future__ import annotations
import pathlib
import struct
import numpy as npy
from typing import Iterable, List, Optional, Union
from autograd_core import View, numpy as np

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
    self._buffer = None
    # offset is required to implement __getitem__
    self.offset = offset

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
      self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
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
      self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
      self.uop = _uop_from_data(flat, self.dtype, self._shape, self._strides)
    elif isinstance(data, bytes):
      if _dtype is None:
        raise ValueError("cannot guess datatype from bytes")
      self.dtype = _dtype
      itemsize = self.dtype.bitsize // 8
      self._shape = (len(data) // itemsize,)
      self._strides = calc_strides(self._shape, self.dtype.bitsize // 8)
      self.uop = _uop_from_data(data, self.dtype, self._shape, self._strides)
    elif isinstance(data, npy.ndarray):
      if data.shape == ():
        data = UOp(op=Ops.CONST, dtype=self.dtype or _from_np_dtypes(data.dtype), arg=(data.tobytes, data.shape, data.strides))
      else:
        self.uop = _uop_from_data(data.tobytes(), dtype=self.dtype or _from_np_dtypes(data.dtype), shape=data.shape, strides=data.strides)
    else:
      raise TypeError(f"unsupported data type: {type(data)!r}")

  @staticmethod
  def from_url(url: str, **kwargs) -> Tensor:
    return Tensor(fetch(url=url), **kwargs)

  def __repr__(self):
    if not self._buffer:
      return f"Tensor <shape={self.shape}, strides={self.strides}>, dtype={self.dtype.name}>"
    else:
      return np(self._buffer)

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

  @staticmethod
  def frombuffer(data: bytes, **kwargs):
    return Tensor(data, **kwargs)

  def __getitem__(self, idx) -> Tensor:
    idx = argfix(idx)
    if len(ellipsis_arr := [i for i,x in enumerate(idx) if x is Ellipsis]) > 1:
      raise ValueError(f"only one ellipsis is possible, provided {len(ellipsis_arr)} ellipses")
    if ellipsis_arr:
      raise NotImplementedError("ellipsis indexing is not supported yet")
    if len(idx) > len(self.shape):
      raise IndexError(f"too many indices for tensor: tensor is {len(self.shape)}-dimensional, but {len(idx)} were indexed")
    index = 0
    new_shape: tuple[int, ...] = tuple()
    new_strides: tuple[int, ...] = tuple()
    new_offset = self.offset
    idx = idx + (slice(None),) * (len(self.shape) - len(idx)) # normalize idx
    for dim in idx:
      if isinstance(dim, int):
        if dim < 0:
          dim += self.shape[index]
        if dim < 0 or dim >= self.shape[index]:
          raise IndexError("index out of bounds")
        new_offset += dim * self.strides[index]
      elif isinstance(dim, slice):
        start,stop,step = dim.indices(self.shape[index])
        if step < 0:
          raise NotImplementedError("negative slice step is not supported yet")
        new_shape += (len(range(start, stop, step)),)
        new_strides += (self.strides[index] * step,)
        new_offset += start * self.strides[index]
      else:
        raise TypeError(f"unsupported index type: {type(dim)!r}")
      index += 1
    view = View(
      new_shape, new_strides, new_offset
    )
    return Tensor(UOp(Ops.SLICE, self.dtype, src=(self.uop,), arg=(view,)), offset=new_offset)

  @classmethod
  def from_np(cls,arr: npy.ndarray) -> Tensor:
    return Tensor(arr)
  
  def __matmul__(self, other: Tensor) -> Tensor:
    if not isinstance(other, Tensor):
      raise ValueError(f"Cannot matmul tensor and {type(other)}")