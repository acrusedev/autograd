from abc import ABC, abstractmethod
from autograd.engine import SharedObject
from ctypes import c_void_p, POINTER, c_int8, c_int, c_char_p

class Allocator(ABC):
  """
  Abstract allocator class defining behaviour of allocators
  """
  def alloc(self, size:int) -> c_void_p: return self._alloc(size)
  def free(self): self._free()
  @abstractmethod
  def _alloc(self, size:int) -> c_void_p: raise NotImplementedError("Allocator must implement _alloc method")
  @abstractmethod
  def _free(self): pass
  @abstractmethod
  def _copyin(self, src: c_char_p, dest: c_void_p, len:int): raise NotImplementedError("Allocator must implement _copyin method")
  @abstractmethod
  def _copyout(): raise NotImplementedError("Allocator must implement _copyout method")

class CPUAllocator(Allocator):
  """
  Cpu allocator allocates memory in RAM for the cpu to access
  """
  def _alloc(self, size:int) -> c_void_p: return SharedObject.alloc_cpu(size)
  def _free(self): pass
  def _copyin(self, src: c_char_p, dest: c_void_p, len:int): SharedObject.copy_in_cpu(src, dest, len)
  def _copyout(): pass