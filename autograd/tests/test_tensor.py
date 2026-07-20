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

  def test_sum_axis(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype='int32')
    row = a.sum(axis=1).realize()
    self.assertEqual(row.shape, (2,))
    self.assertEqual(str(row), str(Tensor([6,15], dtype='int32').realize()))
    col = a.sum(axis=0).realize()
    self.assertEqual(col.shape, (3,))
    self.assertEqual(str(col), str(Tensor([5,7,9], dtype='int32').realize()))

  def test_sum_all(self):
    a = Tensor([[1,2],[3,4]], dtype='int32')
    s = a.sum().realize()
    self.assertEqual(s.shape, ())
    res = Tensor([10], dtype='int32').reshape(()).realize()
    self.assertEqual(str(s), str(res))

  def test_matmul_2d(self):
    a = Tensor([[1,2],[3,4]], dtype='int32')
    b = Tensor([[5,6],[7,8]], dtype='int32')
    # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    c = (a @ b).realize()
    self.assertEqual(c.shape, (2, 2))
    res = Tensor([[19,22],[43,50]], dtype='int32').realize()
    self.assertEqual(str(c), str(res))

  def test_matmul_vec_mat(self):
    a = Tensor([1,2,3], dtype='int32')
    b = Tensor([[1,2],[3,4],[5,6]], dtype='int32')
    # [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    c = (a @ b).realize()
    self.assertEqual(c.shape, (2,))
    res = Tensor([22,28], dtype='int32').realize()
    self.assertEqual(str(c), str(res))

  def test_matmul_mat_vec(self):
    a = Tensor([[1,2,3],[4,5,6]], dtype='int32')
    b = Tensor([1,2,3], dtype='int32')
    # [1+4+9, 4+10+18] = [14, 32]
    c = (a @ b).realize()
    self.assertEqual(c.shape, (2,))
    res = Tensor([14,32], dtype='int32').realize()
    self.assertEqual(str(c), str(res))

  def test_matmul_dot(self):
    a = Tensor([1,2,3], dtype='int32')
    b = Tensor([4,5,6], dtype='int32')
    c = (a @ b).realize()
    self.assertEqual(c.shape, ())
    res = Tensor([32], dtype='int32').reshape(()).realize()
    self.assertEqual(str(c), str(res))

  def test_relu(self):
    a = Tensor([-2, -1, 0, 1, 2], dtype='int32')
    r = a.relu().realize()
    res = Tensor([0, 0, 0, 1, 2], dtype='int32').realize()
    self.assertEqual(str(r), str(res))
