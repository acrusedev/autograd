from autograd.datasets import mnist

train_images, train_labels, test_images, test_labels = mnist()

print(type(train_images))
print(train_images)
