from typing import Callable
from autograd.ops import Ops
from autograd.helpers import calc_strides

def _scalar_shape(op:Ops): return ()
def _shape_from_first_arg(op:Ops): return op.arg[0]
def _shape_from_second_arg(op:Ops): return op.arg[1]
def _shape_from_first_src(op:Ops): return op.src[0].shape
shape_rules: dict[Ops, Callable] = {
    Ops.BUFFER:_shape_from_second_arg,
    Ops.RESHAPE:_shape_from_first_arg,
    Ops.ADD:_shape_from_first_src,
    Ops.CONST:_scalar_shape
}
def _scalar_strides(op:Ops): return ()
def _calc_strides(op:Ops): return calc_strides(op.shape,op.dtype.bitsize//8)
def _strides_from_first_src(op:Ops):return op.src[0].strides
def _strides_from_second_arg(op:Ops):return op.arg[2]
stride_rules = {
    Ops.BUFFER:_strides_from_second_arg,
    Ops.RESHAPE:_calc_strides,
    Ops.ADD:_strides_from_first_src,
    Ops.CONST:_scalar_strides
}
