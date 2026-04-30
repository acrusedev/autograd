from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import List, Tuple, Any, Callable, Dict

from autograd.ops import Ops
from autograd.dtypes import DType
from autograd.helpers import calc_strides

def countOf(t:Iterable, val:int):
  count=0
  for el in t:
    if el==val:count+=1
  return count

def pretty_print(x:UOp, indent=0, cache:dict|None=None) -> str:
  def dfs(x:UOp, cache:dict):
    for child in x.src:
      cache.setdefault(child, [len(cache),0,False])[1] += 1 # sets default to id, refcount, printed
      if cache[child][1] == 1: dfs(child, cache) # check if there are children in children
  if cache is None: dfs(x,cache:={})
  if cache.setdefault(x, [0,0,False])[2]: return f"' ' * {indent}"
  cache.setdefault(x,[0,0,False])[2] = True
  srcs = ''.join(f'\n{pretty_print(src, indent=indent+2, cache=cache)}' for src in x.src)
  return f"{' '*indent}{f'x{cache[x][0]}:=' * (cache[x][1]>1)}{type(x).__name__}({x.op}, {x.dtype}, src=({srcs}))"

# recursive_property replaces functools.cached_property in recursive UOp functions to prevent RecursionError
class recursive_property(property):
  def __init__(self, fxn):
    self.fxn = fxn
    self.nm = "_RECURSIVE_PROPERTY_"+fxn.__name__
    self.__doc__ = fxn.__doc__
  def __get__(self, x:UOp|None, owner=None):
    if x is None: return self
    for node in x.toposort(should_visit=lambda node: self.nm not in node.__dict__):
      node.__dict__[self.nm] = self.fxn(node)
    return x.__dict__[self.nm]

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


@dataclass(frozen=True) # once created is immutable, changing Tensor is possible only by executing new ops
class UOp:
  op: Ops
  dtype: DType # target dtype after operation
  src: Tuple['UOp',...]=tuple()
  arg: Any = None # this depending on the operation will be different data structures

  def __repr__(self):
    return pretty_print(self)

  @staticmethod
  def new_buffer(self):
    pass

  @recursive_property
  def _shape(self):
    """
    for each node in the computation graph this calculates the shape of a tensor that were to be created at that point
    """
    return shape_rules.get(self.op)(self)

  @property
  def shape(self):
    if (ret:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.op} doesn't have a shape")
    return ret

  @recursive_property
  def _strides(self):
    """
    for each node in the computation graph this calculates the shape of a tensor that were to be created at that point
    """
    return stride_rules.get(self.op)(self)

  @property
  def strides(self):
    if (ret:=self._strides) is None: raise RuntimeError(f"strides requested, but {self.op} doesn't have strides")
    return ret

  def toposort(self,should_visit:Callable|None=None) -> dict:
    cache: Dict[UOp, None] = {}
    queue: List[Tuple[UOp, bool]]  = [(self, False)]
    while queue:
      n,v = queue.pop()
      if n in cache: continue
      if not v:
        if should_visit is None or should_visit(n):
          queue.append((n, True))
          for s in reversed(n.src): queue.append((s,False))
      else: cache[n]=None
    return cache
