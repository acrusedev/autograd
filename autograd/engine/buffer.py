import os
from ctypes import CDLL, POINTER, c_uint8, c_int, c_size_t, c_char
from pathlib import Path

class Buffer:
  """
  Representation of an underlying C Tensor structure
  """
  AUTOGRAD_CORE_FILE_PATH = Path(os.getcwd()) / 'libautograd.so' # TODO: shared object needs to compile into dir known at runtime to avoid using getcwd
  def __init__(self):
    self._lib = CDLL(self.AUTOGRAD_CORE_FILE_PATH)
    self._init_buffer_argtypes = [POINTER(c_uint8), c_int, POINTER(c_size_t), POINTER(c_size_t), c_int, c_char]
    self._init_buffer_restype = POINTER(c_uint8)
  def create_buffer(self):
    self._lib.init_buffer.argtypes = self._init_buffer_argtypes
    self._lib.init_buffer.restype = self._init_buffer_restype
