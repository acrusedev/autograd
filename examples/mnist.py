from autograd import Tensor

tensor1 = Tensor([1,2,3,4,5])

tensor2 = tensor1 * 2
print(tensor2.realize())