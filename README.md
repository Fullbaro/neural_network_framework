# Neural Network Framework for Multi and Binary Class Classification and Regression

# Project Description
This project focuses on building a custom neural network framework from scratch for multi-class and binary-class classification tasks, as well as regression. The purpose of the project is primarily educational, aiming to provide a deeper understanding of the underlying mechanisms and operations involved in the functioning of a neural network.

# For examples see [Usage.ipynb](https://github.com/Fullbaro/neural_network_framework/blob/main/Usage.ipynb) 

# Dataset Processing
The notebook provides comprehensive functionality for processing the dataset, including loading, augmentation, balancing, preprocessing, shuffling, normalization, reshaping, and splitting. Moreover, the t-SNE algorithm is applied to visualize the high-dimensional data in a lower-dimensional space, facilitating the exploration of the dataset's structure and relationships.

# Framework Features
The custom-built neural network framework offers several key features:

**Implementation from Scratch:** The framework's design encompasses all aspects of neural networks, including activation functions, optimizers, loss functions, and accuracy metrics. The implementation is done without relying on external libraries such as Keras, TensorFlow, or PyTorch.

**Customizable Model Architecture:** Users have the flexibility to define and customize the architecture of their neural network according to their specific requirements. The framework provides various layer types and activation functions to choose from and build a ***fully connected*** neural network.

**Implemented features:**
- **[Layers](https://github.com/Fullbaro/neural_network_framework/blob/main/lib/layers.py)**
	- Dense
	- Dropout
- **[Activations](https://github.com/Fullbaro/neural_network_framework/blob/main/lib/activations.py)**
	- ReLu
	- Softmax
	- Sigmoid
	- Linear
- **[Optimizers](https://github.com/Fullbaro/neural_network_framework/blob/main/lib/optimizers.py)**
	- SGD
	- Adagrad
	- RMSprop
	- Adam
- **[Loss functions](https://github.com/Fullbaro/neural_network_framework/blob/main/lib/metrics.py)**
	- Categorical crossentropy
	- Binary crossentropy
	- Mean squared error
	- Mean absolute error
- **[Accuracy function](https://github.com/Fullbaro/neural_network_framework/blob/main/lib/metrics.py)**
	- Regression
	- Categorical (both binary and multi class) 


**Training and Evaluation:** The framework supports training the model using the provided dataset and evaluating its performance. Training can be performed with customizable parameters such as the number of epochs, batch size, and early stopping criteria.

**Visualization Capabilities:** The project includes visualizations of the training process, presenting accuracy and loss metrics over epochs. These visualizations provide insights into the model's learning progress and performance.

**Model Persistence:** The framework allows for saving and loading the trained model, including both the model's architecture and its learned weights. This feature facilitates reusability, transfer learning, and further analysis.

# Credits

- [https://smltar.com/dldnn.html](https://smltar.com/dldnn.html)
- [https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

- [https://nnfs.io/](https://nnfs.io/)
- [https://programmathically.com/an-introduction-to-neural-network-loss-functions/](https://programmathically.com/an-introduction-to-neural-network-loss-functions/)
- [https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html](https://www.kdnuggets.com/2020/12/optimization-algorithms-neural-networks.html)
