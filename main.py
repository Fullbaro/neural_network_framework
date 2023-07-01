from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import nnfs
import numpy as np

from layers import *
from activations import *
from losses import *

nnfs.init()




X, y = spiral_data(samples=100, classes=3)
# print("X shape", X.shape)
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()


dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print("Output:", activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)


# TODO: Place this somewhere
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print("Accuracy:", accuracy)
