from autograd.ops import Ops
from autograd.ops.uop import UOp
from autograd.dtypes import least_common_dtype
from autograd.dtypes import as_dtype
from typing import ClassVar
from typing import Self

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
  def __add__(self, other: Self|ConstType) -> Self: # todo: later scheduler should allow adding ints and floats by broadcasting
    # assert isinstance(other, (Self, ConstType)), "can add only a tensor or int or float to a tensor"
    if hasattr(other,'shape'):
      assert self.shape == other.shape, "at this moment broadcasting is not supported, cannot add tensors with different shapes"
      if self.dtype!= other.dtype:
        target_dtype=least_common_dtype(self, other)
      if self.dtype != target_dtype:
        self.uop = UOp(Ops.CAST, dtype=target_dtype, src=(self.uop,))
      if other.dtype != target_dtype:
        other.uop = UOp(Ops.CAST, dtype=target_dtype, src=(other.uop,))
      return self.__class__(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, other.uop)))
    return self.__class__(UOp(Ops.ADD, dtype=least_common_dtype(self, other), src=(self.uop, UOp(Ops.CONST, dtype=as_dtype(other),arg=(other,)))))
