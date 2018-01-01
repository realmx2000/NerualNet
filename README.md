# NeuralNet
Python implementation of a Neural Network API
This project was motivated by an assignment from Stanford's CS229 (Machine Learning) course, where we implemented a simple one layer neural network with sigmoid activations and softmax cross entropy loss to classify the MNIST dataset. The code used for that assignment can be found at http://cs229.stanford.edu/ps/ps4/q1.
The program is a high level neural network API which currently supports ReLU, sigmoid, and tanh activations as well as softmax cross entropy and Weston-Watkins SVM/hinge loss for arbitrary network architectures. The optimizers supported are vanilla gradient descent, gradient descent with momentum, Nesterov accelerated gradient, adagrad, RMSprop, adadelta, and adam.
The last thing I'm looking at trying to implement is batch normalization; once I figure out how to implement the backprop I'll add it.
A sample program which uses the API to classify the MNIST dataset is included in MNIST_test.py.