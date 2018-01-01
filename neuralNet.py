'''
 Created by Zhaoyu Lou on 12/20/17.
'''

import matplotlib.pyplot as plt
import numpy as np

import activations
import loss_functions
import optimizers


# This class provides a python implementation of a fully connected neural network.

class NeuralNet:
    '''
    Initializes some of the hyperparameters of the network.
    layer_sizes is a list containing the number of neurons in each layer.
    input dimension is the dimension of the input data (number of features), and output
    dimension is the number of classes to classify into. Activation should be 'relu' or
    'sigmoid'. Loss should be 'cross entropy' or 'hinge'.
    '''

    def __init__(self, num_hidden_layers, layer_sizes, input_dimension, output_dimension,
                 activation='relu', loss='cross entropy', optimizer='sgd'):
        # Some basic error checking
        assert (activation in ['relu', 'sigmoid', 'tanh']), 'Activation must be \'relu\', \'sigmoid\' or \'tanh\'.'
        assert (loss in ['cross entropy', 'hinge']), 'Loss must be either \'cross entropy\' or \'hinge\'.'
        assert (len(layer_sizes) == num_hidden_layers), 'Too many or too few layer sizes given.'
        assert (optimizer in ['sgd', 'momentum', 'adagrad', 'rmsprop']), \
            'Optimizer must be \'sgd\', \'momentum\', \'adagrad\', \'rmsprop\', or \'adam\'.'

        # Set hyperparameters
        self.num_layers = num_hidden_layers
        self.layer_sizes = layer_sizes
        self.input_dimension = input_dimension
        self.num_classes = output_dimension

        # These two variables are for ease of readability in later functions
        self.w = ['W1']
        self.b = ['b1']
        for layer in range(1, self.num_layers + 1):
            weight = 'W' + str(layer + 1)
            bias = 'b' + str(layer + 1)
            self.w.append(weight)
            self.b.append(bias)

        # Set activation function
        if activation == 'relu':
            self.activation = activations.ReLU()
        elif activation == 'sigmoid':
            self.activation = activations.Sigmoid()
        elif activation == 'tanh':
            self.activation = activations.Tanh()

        # Set loss function
        if loss == 'cross entropy':
            self.loss = loss_functions.CrossEntropy()
        elif loss == 'hinge':
            self.loss = loss_functions.Hinge()

        # Set optimizer
        if optimizer == 'sgd':
            self.optimizer = optimizers.SGD(self.w, self.b)
        elif optimizer == 'momentum':
            self.optimizer = optimizers.Momentum(self.w, self.b)
        elif optimizer == 'adagrad':
            self.optimizer = optimizers.Adagrad(self.w, self.b)
        elif optimizer == 'rmsprop':
            self.optimizer = optimizers.RMSprop(self.w, self.b)
        elif optimizer == 'adam':
            self.optimizer = optimizers.Adam(self.w, self.b)

    # Initializes the architecture of the network and handles some internal book keeping.
    # Returns a dictionary of all weights and biases.
    def init_params(self):
        params = {}

        # Initialize input layer; this is different from other layers because it's shape depends
        # on the shape of the input data.
        params['W1'] = np.random.normal(scale=0.01, size=(self.layer_sizes[0], self.input_dimension))
        params['b1'] = np.random.normal(scale=0.01, size=(self.layer_sizes[0], 1))

        # Initialize hidden layers
        for layer in range(1, self.num_layers):
            params[self.w[layer]] = np.random.normal(size=(self.layer_sizes[layer], self.layer_sizes[layer - 1]))
            params[self.b[layer]] = np.random.normal(size=(self.layer_sizes[layer], 1))

        # Initialize output layer, which depends on the number of classes being classified.
        params[self.w[-1]] = np.random.normal(size=(self.num_classes, self.layer_sizes[self.num_layers - 1]))
        params[self.b[-1]] = np.random.normal(size=(self.num_classes, 1))

        return params

    # Defines the training and dev (validation) datasets. The data and labels follow the standard
    # convention of each row being it's own training example, but my particular implementation
    # is optimized for columns being examples so I transpose the input data.
    def input_data(self, trainData, trainLabels, devData, devLabels):
        self.trainData = trainData.T
        self.trainLabels = (trainLabels.T).astype(int)
        self.devData = devData.T
        self.devLabels = (devLabels.T).astype(int)

    # Forward pass of the neural net, also caches the layer outputs for more efficient backprop.
    # Returns the predictions and the cost, along with the cached values.
    def forward_prop(self, data, labels, params):
        cache = {}

        # Input layer
        z = np.matmul(params['W1'], data) + params['b1']
        cache['1'] = self.activation.activate(z)

        # Hidden layers
        for layer in range(1, self.num_layers):
            z = np.matmul(params[self.w[layer]], cache[str(layer)]) + params[self.b[layer]]
            cache[str(layer + 1)] = self.activation.activate(z)

        # Output layer; no activation function here
        logits = np.matmul(params[self.w[self.num_layers]], cache[str(self.num_layers)]) + params[
            self.b[self.num_layers]]

        # Calculate the loss from logits
        cost, output = self.loss.calculate_loss(logits, labels)

        return cache, output, cost

    # Backwards pass. Calculates gradients of all layers and biases using backpropagation.
    def backward_prop(self, data, cache, labels, params):
        output = params['o']
        deltas = {}

        # Deltas are gradients with respect to layer outputs; it is easy to calculate
        # these using backpropagation.
        deltas[str(self.num_layers + 1)] = self.loss.grad_loss(labels, output)

        for layer in range(self.num_layers, 0, -1):
            gradient = self.activation.gradient(cache[str(layer)])
            deltas[str(layer)] = np.matmul(params[self.w[layer]].T, deltas[str(layer + 1)]) * gradient

        # From the calculated deltas, we can now easily calculate gradients of weights and biases
        # via backprop again.
        grads = {}
        grads['W1'] = np.matmul(deltas['1'], data.T)
        grads['b1'] = np.sum(deltas['1'], axis=1, keepdims=True)
        for layer in range(1, self.num_layers + 1):
            grads[self.w[layer]] = np.matmul(deltas[str(layer + 1)], cache[str(layer)].T)
            grads[self.b[layer]] = np.sum(deltas[str(layer + 1)], axis=1, keepdims=True)

        return grads

    '''
    Trains the neural network.
    Training is done via minibatch gradient descent. Hyperparameters are taken as inputs to this function.
    Note that minibatches are taken sequentially, so the training data must be randomized prior to training.
    The learning rate is reduced by a factor of the decay rate every epoch. The dev set is used for validation 
    and has no effect on the training. If verbose is set to true, the progress is reported every epoch and a 
    plot of the performance history is generated at the end. Returns the set of trained parameters.
    '''

    def nn_train(self, reg_strength, epochs, batch_size, learning_rate, decay_rate, verbose, opt_params):
        # Initialize variables, preprocess data.
        m = self.trainData.shape[1]
        training_losses = []
        dev_losses = []
        training_accs = []
        dev_accs = []
        epoch = 0
        iterations = m // batch_size * epochs
        opt_params['epoch'] = 0

        # Initialize architecture
        params = self.init_params()

        # Make sure the optimization parameters are valid and present
        self.optimizer.verify_params(opt_params)

        # Train model.
        for it in range(iterations):

            # Select minibatches sequentially.
            start = (it * batch_size) % m
            end = start + batch_size
            data = self.trainData[:, start:end]
            labels = self.trainLabels[:, start:end]

            # Forward and backward passes, calculate gradients.
            cache, params['o'], cost = self.forward_prop(data, labels, params)
            grads = self.backward_prop(data, cache, labels, params)

            # Update parameters with both gradients and regularization.
            self.optimizer.update(params, learning_rate, reg_strength, grads, opt_params)

            # Pass current iteration to optimizer in case some parameters need to be corrected
            opt_params['iteration'] = it

            # Decay the learning rate every epoch. and pass the current epoch to the optimizer
            # in case some parameters need to be annealed
            if it % (m // batch_size) == 0:
                epoch += 1
                opt_params['epoch'] = epoch
                learning_rate *= decay_rate

                # If verbose, provide a progress report
                if verbose:
                    # Evaluate model on both the entire training set and the entire dev set.
                    _, _, train_loss = self.forward_prop(self.trainData, self.trainLabels, params)
                    _, _, val_loss = self.forward_prop(self.devData, self.devLabels, params)
                    train_accuracy = self.nn_test(self.trainData, self.trainLabels, params)
                    dev_accuracy = self.nn_test(self.devData, self.devLabels, params)

                    print('Epoch %d, Train Accuracy %f, Validation Accuracy %f' % (epoch, train_accuracy, dev_accuracy))

                    # Store a history of losses and accuracies to plot at the end.
                    training_losses.append(train_loss)
                    dev_losses.append(val_loss)
                    training_accs.append(train_accuracy)
                    dev_accs.append(dev_accuracy)

        # If verbose, plot model performance history.
        if verbose:
            print(len(training_accs))
            self.plot(training_accs, dev_accs, training_losses, dev_losses, epochs)
            plt.show()

        return params

    # Tests the neural network on a validation dataset and returns the accuracy.
    def nn_test(self, data, labels, params):
        _, output, _ = self.forward_prop(data, labels, params)
        accuracy = self.compute_accuracy(output, labels)
        return accuracy

    # Helper function to compute accuracies.
    def compute_accuracy(self, output, labels):
        accuracy = (np.argmax(output, axis=0) == np.argmax(labels, axis=0)).sum() * 1. / labels.shape[1]
        return accuracy

    # Plots the model performance history.
    def plot(self, training_accs, dev_accs, training_losses, dev_losses, epochs):
        plt.plot(range(1, epochs), training_accs[1:])
        plt.xlabel('# of Epochs')
        plt.ylabel('Accuracy')
        plt.title('Neural Network Accuracy')

        plt.plot(range(1, epochs), dev_accs[1:])
        plt.legend(('Training Accuracy', 'Validation Accuracy'))
        plt.figure()

        plt.plot(range(1, epochs), training_losses[1:])
        plt.xlabel('# of Epochs')
        plt.ylabel('Loss')
        plt.title('Neural Network Loss')

        plt.plot(range(1, epochs), dev_losses[1:])
        plt.legend(('Training Loss', 'Validation Loss'))
