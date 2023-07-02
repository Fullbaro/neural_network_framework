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
optimizer = Optimizer_SGD(learning_rate=1, decay=0.001, momentum=0.9)


for epoch in range(10_001):

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
        print(f"Epoch: {epoch}, Accuracy: {accuray}, Loss: {loss}, Learning rate: {optimizer.current_learning_rate}")


    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()