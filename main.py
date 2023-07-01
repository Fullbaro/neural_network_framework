from nnfs.datasets import spiral_data, vertical_data

import matplotlib.pyplot as plt
import nnfs
import numpy as np

from layers import *
from activations import *
from losses import *

nnfs.init()




X, y = vertical_data(samples=100, classes=3)
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()

dense1 = Layer_Dense(2, 3) # Input, Output
activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 2)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# Save values
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


# Try options randomly
for iteration in range(10000):
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    # Traning dat throw layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuray = np.mean(predictions == y)

    if loss < lowest_loss:
        print(f"Network improved. Accuracy: {accuray}, Loss: {loss}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()

        lowest_loss = loss

