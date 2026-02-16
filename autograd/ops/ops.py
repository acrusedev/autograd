from dataclasses import dataclass
from typing import Tuple, Any

from autograd.ops import Ops
from autograd.dtypes import DType

@dataclass(frozen=True)
class UOp:
  op: Ops
  dtype: DType # target dtype after operation
  src: Tuple['UOp',...]=tuple()
  arg: Any = None

  @staticmethod
  def new_buffer(self):
    pass
