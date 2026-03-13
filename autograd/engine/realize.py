from math import prod
from typing import List, Dict, Any, Tuple
from autograd.scheduler import Node
from autograd.engine.allocator import CPUAllocator
from autograd.engine import SharedObject
from autograd.ops import Ops

from ctypes import cast, POINTER, c_int32, c_char_p

def run_schedule(exec_items: List[Node]):
  node_mem_cache: Dict[str, Tuple[Node, Any]]= {}
  for item in exec_items:
    print(item)
    if item.op == Ops.BUFFER:
      size = item.dtype.bitsize//8 * prod(item.args[1])
      cpualloc = CPUAllocator()
      # create memory for the buffer and cache it so that other ops 
      # know where the buffer is written
      mem = cpualloc.alloc(size)
      node_mem_cache[item.id] = (item, mem)
      # write buffer into this memory
      cpualloc._copyin(c_char_p(item.args[0]), mem, size)

    if item.op == Ops.ADD:
      src_mem_addresses = [node_mem_cache.get(x)[1] for x in item.src_ids]
      print(src_mem_addresses)