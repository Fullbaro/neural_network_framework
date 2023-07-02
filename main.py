from nnfs.datasets import spiral_data, vertical_data

import matplotlib.pyplot as plt
import nnfs
import numpy as np

from layers import *
from activations import *
from losses import *
from optimizers import *

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64) # Input, Output
activation1 = Activation_ReLu()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.85)


for epoch in range(10_000):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # print(loss_activation.output[:5])
    # print(loss)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuray = np.mean(predictions==y)
    #print(accuray)

    if not epoch % 100:
        print(f"Epoch: {epoch}, Accuracy: {accuray}, Loss: {loss}")


    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)