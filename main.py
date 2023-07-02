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

X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4) # Input, Output
activation1 = Activation_ReLu()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#optimizer = Optimizer_SGD(learning_rate=1, decay=0.001, momentum=0.9)
#optimizer = Optimizer_Adagrad(decay=1e-4)
#optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)


for epoch in range(10_001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
    loss_activation.loss.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuray = np.mean(predictions==y)

    if not epoch % 100:
        print(f"Epoch: {epoch}\nAccuracy: {accuray}\nLoss: {loss}\nData Loss: {data_loss}\nReg Loss: {regularization_loss}\nLearning rate: {optimizer.current_learning_rate} \n")

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


print("TRAINING DONE")
time.sleep(10)


# Validation

X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')