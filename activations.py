import numpy as np

class Activation_ReLu:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        # Exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #* Kivonás hogy lehogy elszálljon az érték
        # Normalize
        probalities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probalities