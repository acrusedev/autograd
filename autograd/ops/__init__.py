from enum import auto, Enum, IntEnum

class FastEnum(IntEnum):
  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)
  @staticmethod
  def _generate_next_value_(_, __, ___, last_values): return 1 + max([0, *last_values, *[max(c) for c in FastEnum.__subclasses__()]])
class Ops(FastEnum):
  CONST=auto();BUFFER=auto();ADD=auto();RESHAPE=auto()