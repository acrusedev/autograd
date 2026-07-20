from autograd.dtypes import dtypes
import unittest
from autograd import Tensor

class TestTensor(unittest.TestCase):
  def test_can_add_tensors(self):
    a = Tensor([1,2,3,4,5,6,7,8,9])
    b = Tensor([9,8,7,6,5,4,3,2,1])
    e = a+b
    e.realize()
    res = Tensor([10,10,10,10,10,10,10,10,10]).realize()
    self.assertEqual(str(e),str(res))

  def test_can_mul_tensors(self):
    a = Tensor([1,2,3,4], dtype='int32')
    b = Tensor([2,3,4,5], dtype='int32')
    e = (a * b).realize()
    res = Tensor([2,6,12,20], dtype='int32').realize()
    self.assertEqual(str(e), str(res))

  def test_mul_broadcast(self):
    a = Tensor([[1,2,3]], dtype='int32')       # (1, 3)
    b = Tensor([[10],[20],[30]], dtype='int32') # (3, 1)
    e = (a * b).realize()
    self.assertEqual(e.shape, (3, 3))
    res = Tensor([[10,20,30],[20,40,60],[30,60,90]], dtype='int32').realize()
    self.assertEqual(str(e), str(res))

  def test_mul_cast_up(self):
    a = Tensor([1,2,3], dtype='int32')
    b = Tensor([2,3,4], dtype='int64')
    e = (a * b).realize()
    self.assertEqual(e.dtype, dtypes.int64)
    res = Tensor([2,6,12], dtype='int64').realize()
    self.assertEqual(str(e), str(res))

  def test_cast_tensors(self):
    a = Tensor([1,2,3,4,5,6,7,8,9], dtype='int32')
    self.assertEqual(a.dtype, dtypes.int32)
    b = Tensor([1,2,3,4,5,6,7,8,9], dtype='int64')
    self.assertEqual(b.dtype, dtypes.int64)
    e = (a+b).realize()
    self.assertEqual(e.dtype, dtypes.int64)

  def test_select_element(self):
    a = Tensor([1,2,3,4,5,6,7,8,9], dtype='int32')
    b = a[1]
    assert b.dtype == a.dtype
    assert b.shape == ()
    assert b.strides == ()
