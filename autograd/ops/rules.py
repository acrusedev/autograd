from autograd.ops.uop import UOp
from typing import Callable
from autograd.ops import Ops
from autograd.helpers import calc_strides

def _scalar_shape(_): return ()
def _shape_from_first_arg(uop:UOp): return uop.arg[0]
def _shape_from_second_arg(uop:UOp): return uop.arg[1]
def _shape_from_first_src(uop:UOp): return uop.src[0].shape
shape_rules: dict[Ops, Callable] = {
    Ops.BUFFER:_shape_from_second_arg,
    Ops.RESHAPE:_shape_from_first_arg,
    Ops.ADD:_shape_from_first_src,
    Ops.CONST:_scalar_shape
}
def _scalar_strides(_): return ()
def _calc_strides(uop:UOp): return calc_strides(uop.shape,uop.dtype.bitsize//8)
def _strides_from_first_src(uop:UOp):return uop.src[0].strides
def _strides_from_third_arg(uop:UOp):return uop.arg[2]
stride_rules = {
    Ops.BUFFER:_strides_from_third_arg,
    Ops.RESHAPE:_calc_strides,
    Ops.ADD:_strides_from_first_src,
    Ops.CONST:_scalar_strides
}
