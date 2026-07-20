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
  def reshape(self, target_shape, *args) -> Self: ...

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

  def sum(self, axis: int | None = None) -> Self:
    """Sum over one axis (or all axes if axis is None)."""
    if axis is None:
      t: Self = self  # type: ignore[assignment]
      # reduce from the last axis so earlier axis indices stay valid
      for ax in range(len(self.shape) - 1, -1, -1):
        t = t.sum(axis=ax)
      return t
    ndim = len(self.shape)
    ax = axis if axis >= 0 else axis + ndim
    if ax < 0 or ax >= ndim:
      raise ValueError(f"axis {axis} is out of bounds for tensor of dimension {ndim}")
    return self.__class__(UOp(Ops.REDUCE_SUM, dtype=self.dtype, src=(self.uop,), arg=ax))

  def __matmul__(self, other: Self) -> Self:
    """Matrix product via reshape + expand + mul + reduce_sum."""
    if not hasattr(other, 'shape'):
      raise TypeError(f"unsupported operand type(s) for @: '{type(self).__name__}' and '{type(other).__name__}'")
    lhs, rhs = self._cast_pair(other)  # type: ignore[arg-type]
    a_is_1d = len(lhs.shape) == 1
    b_is_1d = len(rhs.shape) == 1
    if a_is_1d:
      lhs = lhs.reshape(1, lhs.shape[0])
    if b_is_1d:
      rhs = rhs.reshape(rhs.shape[0], 1)
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
      raise NotImplementedError(f"matmul currently supports 1D/2D tensors, got {self.shape} @ {other.shape}")
    m, k1 = lhs.shape
    k2, n = rhs.shape
    if k1 != k2:
      raise ValueError(f"shapes {self.shape} and {other.shape} not aligned for matmul: {k1} (dim 1) != {k2} (dim 0)")
    # (m,k) @ (k,n) = sum_k (m,k,1) * (1,k,n) over axis k
    out = (lhs.reshape(m, k1, 1) * rhs.reshape(1, k2, n)).sum(axis=1)
    if a_is_1d and b_is_1d:
      return out.reshape(())
    if a_is_1d:
      return out.reshape(n)
    if b_is_1d:
      return out.reshape(m)
    return out

  def relu(self) -> Self:
    return self.__class__(UOp(Ops.MAX, dtype=self.dtype, src=(self.uop,), arg=0))
