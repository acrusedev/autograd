from typing import Iterable, Optional

from autograd.helpers import fetch

def _infer_shape(data: Iterable) -> Iterable:
    return []

def _check_shape(shape: Iterable, data: Iterable) -> bool:
    """
    Check if provided shape is compatible with existing data.
    All data's elements must fit into the provided shape.
    """
    return True

class Tensor:
    def __init__(self, data: Iterable, shape: Optional[Iterable] = None):
        self.data = data
        self.shape = shape if shape and _check_shape(shape, data) else _infer_shape(data)

    def reshape(self, shape: Iterable):
        self.shape = shape if _check_shape(shape, self.data) else self.shape

    @staticmethod
    def from_url(url: str) -> 'Tensor':
        return Tensor(fetch(url=url))
