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

def loss(predictions, targets):
    # predictions is an array of shape (batch_size, 10) of predictions <0,1> for each number 0-9
    predictions = np.clip(predictions, SMALL, 1 - SMALL) # clip values so that we dont calc log(0) and clip upper val to not drag mean towards any value
    size = len(predictions)
    assert predictions.shape[1] == 10, 'predictions batch must have 10 columns corresponding to probability of each number 0-9'
    correct = predictions[np.arange(size), targets]
    # return -np.log(correct)
    return np.mean(correct)
class Layer:
    def __init__(self, layer_input_size: int, layer_output_size: int, is_last: bool | None = False):
        self.weights = np.random.randn(layer_input_size, layer_output_size).astype(np.float32) * np.sqrt(2.0 / layer_input_size)
        self.biases = np.zeros(layer_output_size).astype(np.float32)
        self.input = None # input is passed in forward pass
        self.outputs = None # outputs calculated during forward pass pre activation function application
        self.z = None # input changed by activation function
        self.is_last = is_last
    def forward(self, inputs: np.ndarray):
        self.outputs = inputs @ self.weights + self.biases
        self.z = relu(self.outputs) if not self.is_last else softmax(self.outputs) # apply softmax to the last dense layer connecting to outputs

class Model:
    def __init__(self):
        self.layer1 = Layer(784, 256)
        self.layer2 = Layer(256, 128)
        self.layer3 = Layer(128, 10)


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = mnist()
    layer1 = Layer(28*28, 256)
    layer2 = Layer(256, 128)
    layer3 = Layer(128,10, is_last=True)

    layer1.forward(train_images / 255.0)
    layer2.forward(layer1.z)
    layer3.forward(layer2.z)
    print("loss ", loss(layer3.outputs, train_labels))
