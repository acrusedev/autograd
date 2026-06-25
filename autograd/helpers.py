import pathlib
import os
import platform
import sys
import hashlib
from typing import Any, TypeVar, Union, Tuple, List, Sequence, TypeGuard, Iterable
import requests
import tempfile
from math import prod
import gzip
from tqdm import tqdm

OSX, WIN = platform.system() == "Darwin", sys.platform == "win32"
cache_dir = pathlib.Path(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))) / "autograd"

def all_values_same(items: Union[Tuple, List]) -> bool:
  return all(x == items[0] for x in items)
def check_shape_compatibility(shape1: tuple[int,...], shape2: tuple[int,...]) -> bool:
  assert not any(x < -1 for x in shape2), "shape dim cannot be lower than -1"
  if prod(shape1)==prod(shape2): return True
  if prod(shape1)%(prod(shape2)*-1)==0 and sum_negatives(shape2) == 1: return True # allow for 1 dimension be equal to -1
  return False

T = TypeVar("T")
def flatten(data) -> Iterable:
  if not hasattr(data, "__len__"): return data
  return [flatten(x) for x in data]

number = float|int
def fully_flatten(data) -> Sequence:
  if hasattr(data, '__len__'):
    result = []
    for item in data:
      result.extend(fully_flatten(item))
    return result
  return [data]

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

def sum_negatives(a: Iterable[int|float]) -> int:
  count = 0
  for el in a:
    if el < 0: count+=1
  return count

def argfix(*x):
  if x and type(x[0]) in (tuple, list):
    if len(x) != 1:
      raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return tuple(x)

def broadcast(shape1: tuple[int, ...], shape2: tuple[int,...]) -> tuple[int,...]:
  val_to_ret = tuple()
  max_len = max(len(shape1), len(shape2))
  for i in range(1, max_len + 1):
    a = shape1[-i] if i <= len(shape1) else 1
    b = shape2[-i] if i <= len(shape2) else 1
    if a == 1 or b == 1:
      val_to_ret += (a,) if a != 1 else (b,)
      continue
    if a == b:
      val_to_ret += (a,)
      continue
    raise ValueError(f"cannot broadcast shapes {shape1} and {shape2}")
  return val_to_ret[::-1]
