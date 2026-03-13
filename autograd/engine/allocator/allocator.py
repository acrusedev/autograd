from abc import ABC, abstractmethod
from autograd.engine import SharedObject
from ctypes import c_void_p

class Allocator(ABC):
  """
  Abstract allocator class defining behaviour of allocators
  """
  def alloc(self, size:int) -> c_void_p: self._alloc(size)
  def free(self): self._free()
  @abstractmethod
  def _alloc(self, size:int) -> c_void_p: raise NotImplementedError("Allocator must implement _alloc method")
  @abstractmethod
  def _free(self): raise NotImplementedError("Allocator must implement _free method")
  @abstractmethod
  def _copyin(): raise NotImplementedError("Allocator must implement _copyin method")
  @abstractmethod
  def _copyout(): raise NotImplementedError("Allocator must implement _copyout method")

class CPUAllocator(Allocator):
  """
  Cpu allocator allocates memory in RAM for the cpu to access
  """
  def _alloc(self, size:int) -> c_void_p: SharedObject.alloc_cpu(size)
