import pathlib
import os
import platform
import sys
import hashlib
from typing import Any, TypeVar, Union, Tuple, List, Sequence, TypeGuard
import requests
import tempfile
import gzip
from tqdm import tqdm

OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"
cache_dir = pathlib.Path(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))) / "autograd"

def all_values_same(items: Union[Tuple, List]) -> bool:
    return all(x == items[0] for x in items)
def check_shape_compatibility(shape1: tuple[int,...], shape2: tuple[int,...]) -> bool:
    mul_res1 = 1
    for element in shape1:
        mul_res1 *= element
    mul_res2 = 1
    for element in shape2:
        mul_res2 *= element
    return mul_res1 == mul_res2

T = TypeVar("T")
def flatten(data: Union[Tuple[T], List[T]]) -> List[T]:
    if not hasattr(data, "__len__"): return [data]
    result = []
    for element in data:
        if hasattr(element, "__len__"):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

def fully_flatten(data: Union[Tuple[T], List[T]]) -> List[T]:
    result = []
    for element in data:
        result.extend(flatten(element))
    return result

def _cache_download_dir() -> pathlib.Path:
    return cache_dir / "downloads"

def fetch(url: str, allow_caching=not os.getenv("DISABLE_HTTP_CACHE"), allow_zipped:bool=False) -> pathlib.Path:
    file = _cache_download_dir() / (hashlib.md5(url.encode('utf-8')).hexdigest() + (".gzip" if allow_zipped else ""))
    if not file.is_file() or not allow_caching:
        (_dir := file.parent).mkdir(parents=True, exist_ok=True)
        # download the file
        with requests.get(url, stream=True, timeout=10) as res:
            assert res.status_code in (200,), res.status_code
            file_size = int(res.headers.get("Content-Length", 0)) if not allow_zipped else None
            progress_bar: tqdm = tqdm(total=file_size, unit="B", unit_scale=True, desc=f"{url.split('/')[-1]}")
            readfile = gzip.GzipFile(fileobj=res.raw) if not allow_zipped else res.raw
            with tempfile.NamedTemporaryFile(dir=_dir, delete=False) as f:
                while chunk := readfile.read(16384):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                f.close()
                pathlib.Path(f.name).rename(file)
            progress_bar.close()
            if file_size and (downloaded_file_size:=os.stat(file).st_size) < file_size: raise RuntimeError(f"fetch size incomplete, {downloaded_file_size} < {file_size}")
            res.close()
    return file

def calc_strides(shape: tuple, itemsize: int) -> tuple:
  if not shape:
      return ()
  strides = [itemsize] * len(shape)
  for i in range(len(shape) - 2, -1, -1): # skip the last element and iterate backwards
      strides[i] = strides[i + 1] * shape[i + 1]
  return tuple(strides)

def all_int(a: Sequence[Any]) -> TypeGuard[Sequence[int]]:
    return all(isinstance(el, int) for el in a)
