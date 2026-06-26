from autograd import Tensor
import unittest


"""
'row expand', lambda: Tensor([[1,2,3]], dtype='int32').expand((2,3))
'vector expand', lambda: Tensor([1,2,3], dtype='int32').expand((2,3))
'scalar-like expand', lambda: Tensor([1], dtype='int32').expand((2,3))
'invalid expand', lambda: Tensor([[1,2,3],[4,5,6]], dtype='int32').expand((3,3))
"""


class TestExpand(unittest.TestCase):
  def test_can_expand(self):
    a = Tensor([1,2,3]).expand(2,3)
    self.assertEqual(a.shape, (2, 3))
    self.assertEqual(a.strides, (0, 8))

    b = Tensor([1,2,3]).expand(3,3)
    self.assertEqual(b.shape, (3, 3))
    self.assertEqual(b.strides, (0, 8))

    c = Tensor([[1,2,3]]).expand(4,3)
    self.assertEqual(c.shape, (4, 3))
    self.assertEqual(c.strides, (0, 8))

    d = Tensor([[1],[2],[3]]).expand(3,4)
    self.assertEqual(d.shape, (3, 4))
    self.assertEqual(d.strides, (8, 0))

  def test_cannot_expand(self):
    with self.assertRaises(ValueError):
      Tensor([1,2]).expand(3,3)

    with self.assertRaises(ValueError):
      Tensor([[1,2,3],[4,5,6]]).expand(3,3)

    with self.assertRaises(ValueError):
      Tensor([[1,2],[3,4]]).expand(2,3)
