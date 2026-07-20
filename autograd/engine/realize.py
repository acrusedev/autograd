from typing import List, Dict, Callable

from autograd_core import View
from autograd.scheduler import Node
from autograd.ops import Ops, view_ops
from autograd_core import Buffer, add_tensors, mul_tensors

def _realize_buffer(item: Node, _cache: Dict[int, Buffer]) -> Buffer:
  return Buffer(
    item.args[0], # type: ignore
    item.args[1], # type: ignore
    item.args[2], # type: ignore
    item.dtype.fmt,
    0
  )

def _realize_add(item: Node, cache: Dict[int, Buffer]) -> Buffer:
  return add_tensors(*[cache[i] for i in item.src_ids])

def _realize_mul(item: Node, cache: Dict[int, Buffer]) -> Buffer:
  return mul_tensors(*[cache[i] for i in item.src_ids])

def _realize_cast(item: Node, cache: Dict[int, Buffer]) -> Buffer:
  return Buffer.cast_buffer(cache[item.src_ids[0]], item.dtype.fmt)

def _realize_view(item: Node, cache: Dict[int, Buffer]) -> Buffer:
  if not isinstance(item.args, View):
    raise ValueError(f"View op received arg that is not a view object {type(item.args)}")
  return cache[item.src_ids[0]].view(item.args)

REALIZERS: Dict[Ops, Callable[[Node, Dict[int, Buffer]], Buffer]] = {
  Ops.BUFFER: _realize_buffer,
  Ops.ADD: _realize_add,
  Ops.MUL: _realize_mul,
  Ops.CAST: _realize_cast,
  **{op: _realize_view for op in view_ops},
}

def run_schedule(exec_items: List[Node]) -> Buffer:
  node_mem_cache: Dict[int, Buffer] = {}
  for item in exec_items:
    realizer = REALIZERS.get(item.op)
    if realizer is None:
      if item.op == Ops.CONST:
        raise NotImplementedError("CONST realize is not implemented yet")
      raise NotImplementedError(f"no realizer for {item.op}")
    node_mem_cache[item.id] = realizer(item, node_mem_cache)
  return node_mem_cache[exec_items[-1].id]
