from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any

from autograd.ops import Ops
from autograd.dtypes import DType

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

@dataclass(frozen=True) # once created is immutable, changing Tensor is possible only by executing new ops
class UOp:
  op: Ops
  dtype: DType # target dtype after operation
  src: Tuple['UOp',...]=tuple()
  arg: Any = None # this depending on the operation will be different data structures

  def __repr__(self):
    return f"UOp <{self.op}, dtype={self.dtype.name}>"

  @staticmethod
  def new_buffer(self):
    pass

  @recursive_property
  def _shape(self):
    """
    for each node in the computation graph this calculates the shape of a tensor that were to be created at that point
    """
    match self.op:
      case Ops.BUFFER:
        _shape=self.arg[1]
      case Ops.RESHAPE:
        _shape=self.arg[0]
    return _shape

  @property
  def shape(self):
    if (ret:=self._shape) is None: raise RuntimeError(f"shape requested, but {self.op} doesn't have a shape")
    return ret

  @recursive_property
  def _strides(self):
    """
    for each node in the computation graph this calculates the shape of a tensor that were to be created at that point
    """
    match self.op:
      case Ops.BUFFER:
        _strides = self.arg[2]
      case Ops.RESHAPE:
        _strides = self.src[0].strides
    return _strides

  @property
  def strides(self):
    if (ret:=self._strides) is None: raise RuntimeError(f"strides requested, but {self.op} doesn't have strides")
    return self._strides()

  def reshape(self, shape: tuple[int,...]) -> 'Tensor':
    # allow only for positive integers except for -1
    if not all(isinstance(element, int) and element >= -1 for element in shape): raise ValueError("only positive integers or -1 are allowed for shape")
    # check if the new shape is compatible with the current shape
    # allow for guesssing one parameter by using -1
    if countOf(shape, -1) > 1: raise ValueError("only one dimension can be -1")
    s = [s for s in shape]
    if -1 in s:
      s[s.index(-1)] = len(self.data) // (-1 * prod(s))
      shape = tuple(s)
    if not check_shape_compatibility(self.shape, shape): raise ValueError(f"new shape {shape} is not compatible with current shape {self.shape}")
    self.shape = shape
    self.strides = calc_strides(shape, self.dtype.bitsize // 8)
    self._buffer.reshape(self.shape, self.strides)
    return self

  def toposort(self, should_visit:Callable|None=None) -> dict[UOp, None]:
    cache: dict[UOp, None] = {}
    stack: list[tuple[UOp, bool]] = [(self, False)] # each stack entry is (node, visited_flag)
    while stack:
      node, visited = stack.pop()
      if node in cache: continue
      if not visited:
        if should_visit is None or should_visit(node):
          stack.append((node, True))  # push node back on stack to process after its srcs
          for s in reversed(node.src): stack.append((s, False)) # push srcs on the stack
      else: cache[node] = None # second time i'm seeing this node, add it to returned toposort
    return cache
