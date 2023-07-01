from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs
import numpy as np

nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Random init
        self.biases = np.zeros((1, n_neurons)) # Init to 0
        print(f"Weights: {self.weights}\nBiases: {self.biases}\n")

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)
# print("X shape", X.shape)
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()


dense = Layer_Dense(2, 3)
dense.forward(X)

print(dense.output[:10])