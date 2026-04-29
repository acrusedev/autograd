from math import prod
from autograd.ops.uop import UOp
from autograd.ops import Ops
from autograd.helpers import check_shape_compatibility
from typing import Self

class MovementMixin:
    def reshape(self, target_shape:list|tuple|int, *args) -> Self:
        if isinstance(target_shape, int):
            if args: target_shape=(target_shape,)+args
            else: target_shape=(target_shape,)
        else: target_shape=tuple(target_shape)
        if target_shape==self._shape: return self # check if Tensor is already os shape target_shape
        if not all(isinstance(element, int) for element in target_shape): raise ValueError("only positive integers or -1 are allowed for shape")
        assert check_shape_compatibility(self._shape, target_shape), f"cannot convert shape {self._shape} to {target_shape}"
        assert target_shape.count(-1) <= 1, "only one -1 dimension is allowed"
        if -1 in target_shape:
            target_shape = list(target_shape)
            target_shape[target_shape.index(-1)] = prod(self.shape) // (-1*prod(target_shape))
            target_shape = tuple(target_shape)
        return self.__class__(UOp(Ops.RESHAPE,dtype=self.dtype, src=(self.uop,), arg=(target_shape,)))