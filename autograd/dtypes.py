from dataclasses import dataclass
from typing import Literal

FmtStr = Literal['f', 'b', 'B'] # later we will extend this, for now we have support for float32 and signed int8
@dataclass(frozen=True) # immutable class object
class DType:
    """
    priority: needed to update tensor's data type when operating on tensors with different dtypes
    name: easy to read for human 
    bit_size: bit size of a single dtype element in memory
    fmt: needed for Python ABI, to parse Rust's formats
    """
    priority: int
    name: str
    bit_size: int
    byte_size: int
    fmt: FmtStr

class dtypes:
    int8 = DType(0, 'int8', 8, 1, 'b')
    uint8 = DType(0, 'uint8', 8, 1, 'B')
    float32 = DType(1, 'f32', 32, 1, 'f')



