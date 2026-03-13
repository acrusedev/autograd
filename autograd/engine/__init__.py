import os
from pathlib import Path
from functools import cached_property
from ctypes import CDLL, c_size_t, c_void_p, c_int, c_uint8, POINTER, c_char_p

class _SharedObject:
  AUTOGRAD_CORE_FILE_PATH = Path(os.getcwd()) / 'libautograd.so' # TODO: shared object needs to compile into dir known at runtime to avoid using getcwd
  @cached_property
  def so(self):
    return CDLL(self.AUTOGRAD_CORE_FILE_PATH)
  def alloc_cpu(self, size: int):
    self.so.allocate_mem_cpu.argtypes = [c_size_t]
    self.so.allocate_mem_cpu.restype = c_void_p
    return self.so.allocate_mem_cpu((c_size_t)(size))
  def copy_in_cpu(self, src: c_char_p, dest: c_void_p, len: int):
    self.so.copy_in_cpu.argtypes = [c_char_p, c_void_p, c_size_t]
    self.so.copy_in_cpu.restype = None
    self.so.copy_in_cpu(src, dest, len)
SharedObject = _SharedObject()
