import os
from ctypes import CDLL
from pathlib import Path

class Buffer:
  """
  Representation of underlying C Tensor structure
  """
  AUTOGRAD_CORE_FILE_PATH = Path(os.getcwd()) / 'libautograd.so'
  def __init__(self):
    self._lib = CDLL(self.AUTOGRAD_CORE_FILE_PATH)
  pass
