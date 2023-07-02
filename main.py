import nnfs
from nnfs.datasets import spiral_data, vertical_data

import matplotlib.pyplot as plt
import numpy as np
import time

from layers import *
from activations import *
from losses import *
from optimizers import *

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1) # Now classes are binary

dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4) # Input, Output
activation1 = Activation_ReLu()
dense2 = Layer_Dense(64, 1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
optimizer = Optimizer_Adam(decay=5e-7)


for epoch in range(10_001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f"Epoch: {epoch}\nAccuracy: {accuracy}\nLoss: {loss}\nData Loss: {data_loss}\nReg Loss: {regularization_loss}\nLearning rate: {optimizer.current_learning_rate} \n")

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()





# Validation

X_test, y_test = spiral_data(samples=1000, classes=2)
y_test = y_test.reshape(-1, 1) # Now classes are binary
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)
# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of
# True/False values, multiplying it by 1 changes it into array
# of 1s and 0s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')