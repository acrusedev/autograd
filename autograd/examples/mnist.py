from autograd.tensor import Tensor

# class Layer:
#   def __init__(self, n_inputs, n_outputs):
#     self.n_inputs = n_inputs
#     self.n_outputs = n_outputs
#     self.biases = Ten
if __name__=="__main__":
#   train_images, train_labels, test_images, test_labels = mnist()
#   print(train_images, train_labels, test_images, test_labels)
  a = Tensor([1,2,3,4,5])
  b = Tensor([1,2,3,4,5,6])
  e = a + b
  e.realize()

