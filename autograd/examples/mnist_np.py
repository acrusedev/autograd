"""
Benchmark model written with numpy to check against autograd's implementation and its speed
"""

import numpy as np
from autograd.helpers import fetch

SMALL = 1e-7

def mnist():
  base_url = "https://raw.githubusercontent.com/fgnt/mnist/master/"
  def _mnist(file): return np.frombuffer(fetch(base_url + file).read_bytes(), np.uint8)
  return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("train-labels-idx1-ubyte.gz")[8:], _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1,28*28)), _mnist("t10k-labels-idx1-ubyte.gz")[8:]

def relu(x):
  return np.maximum(x, 0)

def softmax(x):
  x = x - np.max(x, axis=1, keepdims=True)
  return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# cross entropy
def loss(predictions, targets):
  # predictions is an array of shape (batch_size, 10) of predictions <0,1> for each number 0-9
  predictions = np.clip(predictions, SMALL, 1 - SMALL) # clip values so that we dont calc log(0) and clip upper val to not drag mean towards any value
  size = len(predictions)
  assert predictions.shape[1] == 10, 'predictions batch must have 10 columns corresponding to probability of each number 0-9'
  correct = predictions[np.arange(size), targets]
  return -np.mean(np.log(correct))
class Layer:
  def __init__(self, layer_input_size: int, layer_output_size: int, is_last: bool | None = False):
    self.weights = np.random.randn(layer_input_size, layer_output_size).astype(np.float32) * np.sqrt(2.0 / layer_input_size)
    self.biases = np.zeros(layer_output_size).astype(np.float32)
    self.input = None # input is passed in forward pass
    self.inputs = None
    self.outputs = None # outputs calculated during forward pass pre activation function application
    self.z = None # input changed by activation function
    self.is_last = is_last
  def forward(self, inputs: np.ndarray) -> np.ndarray:
    self.inputs = inputs
    self.outputs = inputs @ self.weights + self.biases
    self.z = relu(self.outputs) if not self.is_last else softmax(self.outputs) # apply softmax to the last dense layer connecting to outputs
    return self.z
  def backprop(self,dy, targets=None): # dy-derivative of activation function from the previous layer
    """
    dy - gradient from the next layer
    targets - only for the last layer softmax+cross_entropy
    """
    batch_size = self.inputs.shape[0]
    if self.is_last:
      # use softmax derivative
      dz = self.z.copy()
      dz[np.arange(batch_size), targets] -= 1
      dz = dz / batch_size  # średnia po batch
    else:
      # use relu derivative
      # derivative of relu function
      """
      Calculating gradients
      derivative with regards to inputs:
      z'(i) = weights[] * drelu
      derivative with regards to weights:
      z'(w) = inputs[] * drelu
      derivative with regards to bias:
      z'(b) = 1 * drelu
      """
      drelu = (self.outputs>0).astype(float)
      dz = dy*drelu

    self.dweights = np.dot(self.inputs.T, dz) # we transpose inputs since we are going backwards
    self.dbiases = np.sum(dz,axis=0,keepdims=False)
    self.dinputs = np.dot(dz, self.weights.T) # same here
    return self.dinputs
class Model:
  def __init__(self):
    self.layer1 = Layer(784, 256)
    self.layer2 = Layer(256, 128)
    self.layer3 = Layer(128, 10, is_last=True)

  def forward(self, inputs):
    z1 = self.layer1.forward(inputs)
    z2 = self.layer2.forward(z1)
    z3 = self.layer3.forward(z2)
    return z3

  def backprop(self, targets):
    d3 = self.layer3.backprop(None, targets=targets)
    d2 = self.layer2.backprop(d3)
    d1 = self.layer1.backprop(d2)

  def update_weights(self, learning_rate):
    self.layer1.weights -= learning_rate * self.layer1.dweights
    self.layer1.biases -= learning_rate * self.layer1.dbiases
    self.layer2.weights -= learning_rate * self.layer2.dweights
    self.layer2.biases -= learning_rate * self.layer2.dbiases
    self.layer3.weights -= learning_rate * self.layer3.dweights
    self.layer3.biases -= learning_rate * self.layer3.dbiases

  def train(self, train_images, train_labels, epochs, learning_rate, batch_size=64):
    train_images = train_images / 255.0
    train_dataset_size = train_images.shape[0]

    for epoch in range(epochs):
      train_images_indexes = np.arange(train_dataset_size)
      np.random.shuffle(train_images_indexes) # shuffle train_images in memory
      train_images = train_images[train_images_indexes]
      train_labels = train_labels[train_images_indexes]

      epoch_loss = 0
      epoch_acc = 0
      num_batches = train_dataset_size // batch_size

      for i in range(0, train_dataset_size, batch_size):
        batch_images = train_images[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]

        if len(batch_labels) == 0: continue

        predictions = self.forward(batch_images)
        ce_loss = loss(predictions, batch_labels)
        train_acc = self.accuracy(predictions, batch_labels)

        self.backprop(batch_labels)
        self.update_weights(learning_rate)

        epoch_loss += ce_loss
        epoch_acc += train_acc

      avg_loss = epoch_loss / num_batches
      avg_acc = epoch_acc / num_batches
      print(f"epoch {epoch+1}/{epochs} - ce_loss: {avg_loss}, accuracy: {avg_acc}")
  def accuracy(self, predictions, targets):
      predicted_classes = np.argmax(predictions, axis=1)
      correct = np.sum(predicted_classes == targets)
      return correct / len(targets)

  def predict(self, images):
    images = images / 255.0
    predictions = self.forward(images)
    return np.argmax(predictions, axis=1)


if __name__ == "__main__":
  train_images, train_labels, test_images, test_labels = mnist()

  model = Model()
  model.train(train_images, train_labels, 10, 0.1)
  sample_predictions = model.predict(test_images[:100])
