from typing import Dict
from autograd.dtypes import DType
from autograd.ops import Ops
from autograd.ops.uop import UOp

def _create_nodes_from_toposort(d:Dict[UOp,None]):
    key_index = {}
    nodes = []
    order: list[UOp] = []
    for i,k in enumerate(d.keys()):
        key_index[k]=i
        order.append(k)

    for el in order:
        nodes.append(
            Node(key_index[el], el.op, el.dtype, el.shape, el.strides, src_ids=tuple([key_index[k] for k in el.src]),args=el.arg)
        )

    return nodes


class Scheduler:
    """
    scheduler should prepare based on ops the plan for linealizer on how to most efficiently schedule operations
    """
    def __init__(self,uop: UOp):
        self.nodes = _create_nodes_from_toposort(uop.toposort())

class Node:
    def __init__(self, id: int,op: Ops, dtype: DType, shape: tuple, strides: tuple,src_ids: tuple[int,...], args:tuple|None=None):
        self.id=id
        self.op=op
        self.dtype=dtype
        self.shape=shape
        self.strides=strides
        self.src_ids=src_ids
        self.args=args

