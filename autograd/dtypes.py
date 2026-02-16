from dataclasses import dataclass
from typing import Literal, Final

FmtStr = Literal['f', 'b', 'B', 'd', 'h', 'q', 'e', 'v', 'i', '?'] # later we will extend this, for now we have support for float32 and signed int8

class DTypeMetaclass(type):
    """
    Metaclass for DType is a singleton.
    We need singleton here so that when comparing dtypes: dtype.float32 is dtype.float32 # True
    singleton per attribute set pattern ensures that only one instance of DType is created for each unique set of attributes
    """
    dcache:dict[tuple,"DType"] = {}
    def __call__(cls, *args, **kwargs):
        """
        args here is a tuple of (priority, name, bit_size, count, fmt)
        """
        if (ret:=DTypeMetaclass.dcache.get(args)) is not None: return ret
        DTypeMetaclass.dcache[args] = ret = super().__call__(*args)
        return ret

@dataclass(frozen=True, eq=False) # immutable class object
class DType(metaclass=DTypeMetaclass):
    priority: int # this determines upcasting behavior
    name: str
    bitsize: int
    count: int
    fmt: FmtStr
    @staticmethod
    def new(priority:int, bitsize:int, name:str, fmt:FmtStr): return DType(priority, name, bitsize, 1, fmt)

class dtypes:
    # https://docs.python.org/3/library/struct.html#struct-format-strings
    boolean: Final[DType] = DType.new(0, 8, 'bool','?')
    int8: Final[DType] = DType.new(1, 8, 'int8','b')
    uint8: Final[DType] = DType.new(2, 8, 'uint8', 'B')
    int16: Final[DType] = DType.new(3, 16, 'int16', 'h')
    int32: Final[DType] = DType.new(4, 32, 'int32', 'i')
    int64: Final[DType] = DType.new(5, 64, 'int64', 'q')
    bfloat16: Final[DType] = DType.new(6, 16, 'bfloat16', 'v') # requires custom implementation
    float16: Final[DType] = DType.new(7, 16, 'float16', 'e')
    float32: Final[DType] = DType.new(8, 32, 'f32', 'f')
    float64: Final[DType] = DType.new(9, 64, 'f64', 'd')


    @staticmethod
    def infer_dtype(x) -> DType:
        if not hasattr(x,'__len__'):
            # scalar value
            if isinstance(x, float): return dtype_default_float
            if isinstance(x, int) and not isinstance(x, bool): return dtype_default_int
            if isinstance(x, bool): return dtypes.boolean
        else:
            # typing.Iterable
            if hasattr(x,'dtype'):return x.dtype
            if len(x) == 0: return dtype_default_int
            for element in x[0:100]:
                if isinstance(element,float):return dtype_default_float
            return dtype_default_int

dtype_default_float = dtypes.float32
dtype_default_int = dtypes.int32

DTypeLike=str|DType
def to_dtype(dtype:DTypeLike) -> DType: return dtype if isinstance(dtype, DType) else getattr(DType,dtype)
