from math import prod
from typing import List, Dict, Any, Tuple
from autograd.scheduler import Node
from autograd.engine.allocator import CPUAllocator
from autograd.engine import SharedObject
from autograd.ops import Ops

def run_schedule(exec_items: List[Node]):
  node_mem_cache: Dict[str, Tuple[Node, Any]]= {}
  for item in exec_items:
    if item.op == Ops.BUFFER:
      size = item.dtype.bitsize//8 * prod(item.args[1])
      mem = SharedObject.alloc_cpu(size)
      node_mem_cache[item.id] = (item, mem)
