import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Random init
        self.biases = np.zeros((1, n_neurons)) # Init to 0
        #print(f"Weights: {self.weights}\nBiases: {self.biases}\n")

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases