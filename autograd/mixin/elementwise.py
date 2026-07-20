from __future__ import annotations
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

  def _broadcasted(self, other: Self) -> tuple[Self, Self]:
    if self.shape == other.shape:
      return self, other
    target_shape = broadcast_shape(self.shape, other.shape)
    return self.expand(target_shape), other.expand(target_shape)

  def _cast_pair(self, other: Self) -> tuple[Self, Self]:
    if self.dtype == other.dtype:
      return self, other
    target_dtype = least_common_dtype(self, other)
    lhs, rhs = self, other
    if lhs.dtype != target_dtype:
      lhs = self.__class__(UOp(Ops.CAST, dtype=target_dtype, src=(lhs.uop,)))
    if rhs.dtype != target_dtype:
      rhs = self.__class__(UOp(Ops.CAST, dtype=target_dtype, src=(rhs.uop,)))
    return lhs, rhs

  def _binop(self, other: Self|ConstType, op: Ops) -> Self:
    if hasattr(other, 'shape'):
      lhs, rhs = self._broadcasted(other)  # type: ignore[arg-type]
      lhs, rhs = lhs._cast_pair(rhs)
      return self.__class__(UOp(op, dtype=least_common_dtype(lhs, rhs), src=(lhs.uop, rhs.uop)))
    # scalar RHS: promote via CONST (realize for CONST is not wired yet)
    return self.__class__(UOp(op, dtype=least_common_dtype(self, other), src=(self.uop, UOp(Ops.CONST, dtype=as_dtype(other), arg=(other,)))))

  def __add__(self, other: Self|ConstType) -> Self:
    return self._binop(other, Ops.ADD)

  def __mul__(self, other: Self|ConstType) -> Self:
    return self._binop(other, Ops.MUL)
