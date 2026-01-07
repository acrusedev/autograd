from autograd.datasets import mnist

train_images, train_labels, test_images, test_labels = mnist()

print(train_images, train_labels, test_images, test_labels)
