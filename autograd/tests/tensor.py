import unittest
from autograd import Tensor

class TestTensor(unittest.TestCase):
    def test_can_add_tensors(self):
        a = Tensor([1,2,3,4,5,6,7,8,9])
        b = Tensor([9,8,7,6,5,4,3,2,1])
        e = a+b
        e.realize()
        res = Tensor([10,10,10,10,10,10,10,10,10]).realize()
        self.assertEqual(e.shape,a.shape)
        self.assertEqual(str(e),str(res))
