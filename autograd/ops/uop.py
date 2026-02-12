import typing
from autograd.ops.ops import Ops
from autograd import Tensor
from autograd.dtypes import DType

class UOp:
    def __init__(self, op:Ops, src:typing.Tuple[UOp, ...], dtype:typing.Optional[DType]=None,*args):
        pass
    def __repr__(self):
        return ""