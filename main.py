from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs
import numpy as np

nnfs.init()

# class Layer_Dense:

#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Random init
#         self.biases = np.zeros((1, n_neurons)) # Init to 0
#         print(f"Weights: {self.weights}\nBiases: {self.biases}\n")

#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

# class Activation_ReLu:

#     def forward(self, inputs):
#         self.output = np.maximum(0, inputs)


# X, y = spiral_data(samples=100, classes=3)
# # print("X shape", X.shape)
# # plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# # plt.show()


# dense = Layer_Dense(2, 3)
# activation = Activation_ReLu()

# dense.forward(X)
# activation.forward(dense.output)

# print(activation.output[:5])


layer_outputs = [4.8, 1.21, 2.385]

# Exponentaiate
exp_values = np.exp(layer_outputs)
print("Exp values:", exp_values)

# Normalize
norm_values = exp_values / np.sum(exp_values)
print("Normalized:", norm_values)
print("Sum of normalized values:", np.sum(norm_values))


# For batches
inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
]
exp_values = np.exp(inputs)
probalities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis=1 oszplo menti
print("Probalities:", probalities)