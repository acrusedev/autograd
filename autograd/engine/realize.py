from autograd.dtypes import as_dtype
from autograd.dtypes import dtypes
from autograd.helpers import calc_strides
from typing import List, Dict, Any, Tuple
from autograd.scheduler import Node
from autograd.ops import Ops
from autograd_core import Buffer
from autograd_core import add_tensors, select_buffer_element


def run_schedule(exec_items: List[Node]) -> Buffer:
  node_mem_cache: Dict[int, Buffer]= {}
  for item in exec_items:
    print(item)
    if item.op == Ops.BUFFER:
      buffer = Buffer(
        item.args[0], # type: ignore
        item.args[1], # type: ignore
        item.args[2], # type: ignore
        item.dtype.fmt,
        0
      )
      node_mem_cache[item.id] = buffer

    if item.op == Ops.ADD:
      buffers = [node_mem_cache[id] for id in item.src_ids]
      node_mem_cache[item.id] = add_tensors(*buffers)
    if item.op == Ops.CONST:
      pass
    if item.op == Ops.RESHAPE:
      pass
    if item.op == Ops.CAST:
      b = node_mem_cache[item.src_ids[0]]
      node_mem_cache[item.id] = Buffer.cast_buffer(b, item.dtype.fmt)
    if item.op == Ops.SELECT:
      b = node_mem_cache[item.src_ids[0]]
      node_mem_cache[item.id] = select_buffer_element(b, item.args[0])
    if item.op == Ops.SLICE:
      raise NotImplementedError("SOON tm")

  return node_mem_cache[exec_items[-1].id]