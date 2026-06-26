import unittest
from autograd.ops.uop import broadcast_shape

"""
broadcast((3, 1), (1, 4)) == (3, 4)
broadcast((5, 3, 1), (3, 4)) == (5, 3, 4)
broadcast((2, 3), (3,)) == (2, 3)
broadcast((3,), (2, 3)) == (2, 3)
broadcast((), (2, 3)) == (2, 3)
broadcast((1,), (2, 3)) == (2, 3)
should raise:
broadcast((2, 3), (4,))
broadcast((2, 3), (2, 4))
"""


class TestBroadcast(unittest.TestCase):
  def test_can_broadcast(self):
    self.assertEqual(broadcast_shape((3,1), (1,4)), (3,4))
    self.assertEqual(broadcast_shape((5,3,1), (3,4)), (5,3,4))
    self.assertEqual(broadcast_shape((2,3), (3,)), (2,3))
    self.assertEqual(broadcast_shape((), (2,3)), (2,3))
    self.assertEqual(broadcast_shape((1,), (2,3)), (2,3))

    with self.assertRaises(ValueError):
      broadcast_shape((2,3),(4,))

    with self.assertRaises(ValueError):
      broadcast_shape((2,3),(2,4))