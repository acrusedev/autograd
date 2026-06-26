from typing import List, Dict

from autograd_core import View
from autograd.scheduler import Node
from autograd.ops import Ops
from autograd_core import Buffer, add_tensors

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
      buffers = [node_mem_cache[i] for i in item.src_ids]
      node_mem_cache[item.id] = add_tensors(*buffers)
    if item.op == Ops.CONST:
      pass
    if item.op == Ops.RESHAPE:
      buffer = node_mem_cache[item.src_ids[0]]
      if isinstance(item.args, View):
        node_mem_cache[item.id] =  buffer.view(item.args)
      else:
        raise ValueError(f"View op received arg that is not a view object {type(item.args)}")
    if item.op == Ops.CAST:
      b = node_mem_cache[item.src_ids[0]]
      node_mem_cache[item.id] = Buffer.cast_buffer(b, item.dtype.fmt)
    if item.op == Ops.SLICE:
      buffer = node_mem_cache[item.src_ids[0]]
      if isinstance(item.args, View):
        node_mem_cache[item.id] = buffer.view(item.args)
      else:
        raise ValueError(f"View op received arg that is not a view object {type(item.args)}")
    if item.op == Ops.EXPAND:
      print("Expanding")
      buffer = node_mem_cache[item.src_ids[0]]
      if isinstance(item.args, View):
        node_mem_cache[item.id] = buffer.view(item.args)
      else:
        raise ValueError(f"View op received arg that is not a view object {type(item.args)}")

  return node_mem_cache[exec_items[-1].id]