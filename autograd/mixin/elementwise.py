from __future__ import annotations
from math import prod
from typing import ClassVar
from typing import Self
from autograd.ops import Ops
from autograd.ops.uop import UOp
from autograd.dtypes import least_common_dtype
from autograd.dtypes import as_dtype
from autograd.ops.uop import broadcast_shape

class InvalidType:
  _instance: ClassVar[InvalidType|None] = None
  def __new__(cls):
    if cls._instance is None: cls._instance = object.__new__(cls)
    return cls._instance
  def __eq__(self, other): return self is other
  def __lt__(self, other): return self is not other
  def __gt__(self, other): return self is not other
  def __hash__(self): return id(self)
  def __repr__(self): return "Invalid"
  def __reduce__(self): return (InvalidType, ())  # unpickle returns the singleton
  def __format__(self, spec): return "Invalid" 

PyConst = float|int|bool
ConstType = PyConst|InvalidType

class ElementwiseMixin:
  @property
  def shape(self):
    raise NotImplementedError
  @property
  def strides(self):
    raise NotImplementedError
  @property
  def dtype(self):
    raise NotImplementedError
  def expand(self, target_shape: int|tuple[int,...], *args: int) -> Self: ...
  def __add__(self, other: Self|ConstType) -> Self: # todo: later scheduler should allow adding ints and floats by broadcasting
    if hasattr(other,'shape'):
      # assert self.shape == other.shape, f"at this moment broadcasting is not supported, cannot add tensors with different shapes {self.shape} != {other.shape}"
      if self.shape != other.shape:
        target_shape = broadcast_shape(self.shape, other.shape)
        self = self.expand(target_shape)
        other = other.expand(target_shape)
      if self.dtype!= other.dtype:
        target_dtype=least_common_dtype(self, other)
        if self.dtype != target_dtype:
          self.uop = UOp(Ops.CAST, dtype=target_dtype, src=(self.uop,))
        if other.dtype != target_dtype:
          other.uop = UOp(Ops.CAST, dtype=target_dtype, src=(other.uop,))
      return self.__class__(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, other.uop)))
    return self.__class__(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, UOp(Ops.CONST, dtype=as_dtype(other),arg=(other,)))))
