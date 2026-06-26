from autograd.helpers import argfix
from autograd.helpers import calc_strides
from autograd.dtypes import DType
from autograd_core import View
from math import prod
from autograd.ops.uop import UOp
from autograd.ops import Ops
from autograd.helpers import check_shape_compatibility
from typing import Self

class MovementMixin:
  @property
  def shape(self) -> tuple[int,...]:
    raise NotImplementedError
  @property
  def strides(self) -> tuple[int,...]:
    raise NotImplementedError
  @property
  def offset(self) -> int:
    raise NotImplementedError
  @property
  def dtype(self) -> DType:
    raise NotImplementedError

  def reshape(self, target_shape:list|tuple|int, *args) -> Self:
    if isinstance(target_shape, int):
      if args: target_shape=(target_shape,)+args
      else: target_shape=(target_shape,)
    else: target_shape=tuple(target_shape)
    if target_shape==self.shape: return self # check if Tensor is already os shape target_shape
    if not all(isinstance(element, int) for element in target_shape): raise ValueError("only positive integers or -1 are allowed for shape")
    assert check_shape_compatibility(self.shape, target_shape), f"cannot convert shape {self._shape} to {target_shape}"
    assert target_shape.count(-1) <= 1, "only one -1 dimension is allowed"
    if -1 in target_shape:
      target_shape = list(target_shape)
      target_shape[target_shape.index(-1)] = prod(self.shape) // (-1*prod(target_shape))
      target_shape = tuple(target_shape)
    new_strides = calc_strides(target_shape, self.dtype.bitsize//8)
    new_view = View(target_shape, new_strides, self.offset)
    return self.__class__(UOp(Ops.RESHAPE,dtype=self.dtype, src=(self.uop,), arg=new_view))
  def expand(self, target_shape: int|tuple[int,...], *args: int):
    if args:
      if not isinstance(target_shape, int): raise ValueError("Error: expand(2,3) or expand((2,3))")
      target_shape = (target_shape,) + args
    new_strides = []
    for i in range(1, len(target_shape) + 1):
      a = self.shape[-i] if i <= len(self.shape) else 1
      b = target_shape[-i] if i <= len(target_shape) else 1
      if a==b:
        if i <= len(self.shape):
          new_strides += [self.strides[-i]]
        else:
          new_strides += [0]
      elif a==1:
        new_strides += [0]
      else:
        raise ValueError(f"cannot expand tensor with shape {self.shape} with target_shape {target_shape}")
    new_strides = tuple(reversed(new_strides))
    view = View(target_shape, new_strides, self.offset)
    return self.__class__(UOp(Ops.EXPAND, dtype=self.dtype, src=(self.uop,), arg=view))

  def __getitem__(self,idx) -> Self:
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
    view = View(new_shape, new_strides, new_offset)
    return self.__class__(UOp(Ops.SLICE, self.dtype, src=(self.uop,), arg=view))