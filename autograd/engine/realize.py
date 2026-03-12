import os, struct
from ctypes import CDLL, POINTER, c_uint8, c_int, c_size_t, c_char
from pathlib import Path

if __name__=='__main__':
  lib_file_path = Path(os.getcwd()) / 'libautograd.so'
  lib = CDLL(lib_file_path)
  print(lib)
  lib.init_buffer.argtypes = [POINTER(c_uint8), c_int, POINTER(c_size_t), POINTER(c_size_t), c_int, c_char]
  lib.init_buffer.restype = POINTER(c_uint8)
  # dla Tensora [1,2,3]
  raw_bytes = struct.pack('3b', 1,2,3)
  a = lib.init_buffer((c_uint8*1).from_buffer_copy(raw_bytes), 3, (c_size_t*1)(3), (c_size_t*1)(3), 3, b'b')
  print(a)
