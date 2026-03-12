from abc import ABC, abstractmethod

class Allocator(ABC):
  @abstractmethod
  def _alloc(self): raise NotImplementedError("Allocator must implement _alloc method")
  @abstractmethod
  def _free(self): raise NotImplementedError("Allocator must implement _free method")
  @abstractmethod
  def _copyin(): raise NotImplementedError("Allocator must implement _copyin method")
  @abstractmethod
  def _copyout(): raise NotImplementedError("Allocator must implement _copyout method")