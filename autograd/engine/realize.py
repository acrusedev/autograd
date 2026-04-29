from math import prod
from typing import List, Dict, Any, Tuple
from autograd.scheduler import Node
from autograd.ops import Ops
from autograd_core import Buffer


def run_schedule(exec_items: List[Node]):
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
      print(f"BUFFER {buffer}")

    # if item.op == Ops.ADD:
    #   src_mem_addresses = [node_mem_cache.get(x)[1] for x in item.src_ids]
    #   print(src_mem_addresses)