from typing import Iterable, Optional

def _infer_shape(data: Iterable) -> Iterable:
    pass

def _check_shape(shape: Iterable, data: Iterable) -> bool:
    """
    Check if provided shape is compatible with existing data.
    All data's elements must fit into the provided shape.
    """
    pass

class Tensor:
    def __init__(self, data: Iterable, shape: Optional[Iterable] = None):
        self.data = data
        self.shape = shape if _check_shape(shape, data) and shape is not None else _infer_shape(data)

    def reshape(self, shape: Iterable):
        self.shape = shape if _check_shape(shape, self.data) else self.shape

    @staticmethod
    def from_url(url: str) -> 'Tensor':
        return Tensor((0,))
