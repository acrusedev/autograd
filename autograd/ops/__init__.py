from enum import auto, Enum, IntEnum

class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])
class Ops(FastEnum):
  """
   entry point to computation graph. the buffer operation should carry src with binary data
   and lazydata: (shape,strides)
  """
  BUFFER=auto()
  RESHAPE=auto()
  ADD=auto() # lazydata:()
  CONST=auto() # change ints, floats into a uop
  CAST=auto() # lazydata: (tensor, shape)
  SLICE=auto()
  EXPAND=auto()

"""
View operations do not run any compute on the underlying data. They only change the way the underlying data is interpreted.
All view ops take View object as an arg
"""
view_ops = [Ops.RESHAPE, Ops.SLICE, Ops.EXPAND] # arg=View(shape, strides, offset)
unary_ops = [Ops.CAST]
binary_ops = [Ops.ADD] # src=(Tensor, Tensor)
compute_ops = unary_ops + binary_ops
input_ops = [Ops.BUFFER, Ops.CONST]
