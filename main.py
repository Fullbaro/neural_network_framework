import nnfs
from nnfs.datasets import *

import matplotlib.pyplot as plt
import numpy as np
import time

from layers import *
from activations import *
from metrics import *
from optimizers import *
from model import *

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model()

model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())


model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-7),
    accuracy=Accuracy_Regression()
)

model.finalize()

model.train(
    X,
    y,
    validation_data=(X_test, y_test),
    epoch=10_000,
    print_every=100
)