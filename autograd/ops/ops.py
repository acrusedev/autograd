from enum import auto, Enum, IntEnum

class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])
class Ops(FastEnum):
  #unary operations
  EXP2=auto();LOG2=auto();SIN=auto();SQRT=auto();NEG=auto();CAST=auto();RECIPROCAL=auto()
  #binary operations
  ADD=auto();SUB=auto();MUL=auto();DIV=auto();MAX=auto();MAX=auto();MOD=auto();CMPLT=auto()
  #reduce ops
  SUM=auto();MAX=auto()
  #movement ops
  RESHAPE=auto();PERMUTE=auto();EXPAND=auto();PAD=auto();SHRINK=auto()
  #global ops
  LOAD=auto();STORE=auto();CONST=auto()