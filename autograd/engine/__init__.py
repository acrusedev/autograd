import os
from pathlib import Path
from functools import cached_property
from ctypes import CDLL, c_size_t, c_void_p

class _SharedObject:
  AUTOGRAD_CORE_FILE_PATH = Path(os.getcwd()) / 'libautograd.so' # TODO: shared object needs to compile into dir known at runtime to avoid using getcwd
  @cached_property
  def so(self):
    return CDLL(self.AUTOGRAD_CORE_FILE_PATH)
  def alloc_cpu(self, size: int):
    self.so.allocate_mem_cpu.argtypes = [c_size_t]
    self.so.allocate_mem_cpu.restype = c_void_p
    self.so.allocate_mem_cpu((c_size_t)(size))
SharedObject = _SharedObject()
