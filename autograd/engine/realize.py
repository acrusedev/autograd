from typing import List, Dict, Any, Tuple
from autograd.scheduler import Node
from autograd.ops import Ops
from autograd_core import Buffer
from autograd_core import add_tensors


def run_schedule(exec_items: List[Node]) -> Buffer:
  node_mem_cache: Dict[str, Tuple[Node, Any]]= {}
  for item in exec_items:
    print(item)
    if item.op == Ops.BUFFER:
      buffer = Buffer(
        item.args[0],
        item.args[1],
        item.args[2],
        item.dtype.fmt
      )
      node_mem_cache[item.id] = buffer

    if item.op == Ops.ADD:
      buffers = [node_mem_cache.get(id) for id in item.src_ids]
      node_mem_cache[item.id] = add_tensors(*buffers)
    if item.op == Ops.CONST:
      pass
    if item.op == Ops.RESHAPE:
      pass
  return node_mem_cache[exec_items[-1].id]