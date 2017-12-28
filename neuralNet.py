import numpy as np
import matplotlib.pyplot as plt

'''
This class provides a python implementation of a fully connected neural network API
using sigmoid or relu nonlinearities and softmax cross entropy loss. 
'''
class NeuralNet:

    '''
    Initializes some of the hyperparameters of the network.
    layer_sizes is a list containing the number of neurons in each layer.
    input dimension is the dimension of the input data (number of features), and output
    dimension is the number of classes to classify into. Activation should be 'relu' or
    'sigmoid'.
    '''
    def __init__(self, num_hidden_layers, layer_sizes, input_dimension, output_dimension, activation, loss):
        #Some basic error checking
        assert(activation in ['relu', 'sigmoid']),'Activation must be either \'relu\' or \'sigmoid\'.'
        assert (loss in ['cross entropy', 'hinge']), 'Loss must be either \'cross entropy\' or \'hinge\'.'
        assert(len(layer_sizes) == num_hidden_layers), 'Too many or too few layer sizes given.'

        self.num_layers = num_hidden_layers
        self.layer_sizes = layer_sizes
        self.input_dimension = input_dimension
        self.num_classes = output_dimension
        self.active_type = activation
        self.loss = loss

    #Initializes the architecture of the network and handles some internal book keeping.
    #Returns a dictionary of all weights and biases.
    def init_params(self):
        params = {}

        #These two variables are just for ease of readability in later functions.
        self.w = ['W1']
        self.b = ['b1']

        #Initialize input layer; this is different from other layers because it's shape depends
        #on the shape of the input data.
        params['W1'] = np.random.normal(size=(self.layer_sizes[0], self.input_dimension))
        #params['b1'] = np.zeros((self.layer_sizes[0], 1))
        params['b1'] = np.random.normal(size=(self.layer_sizes[0], 1))

        #Initialize hidden layers
        for layer in range(1, self.num_layers):
            weight = 'W' + str(layer + 1)
            bias = 'b' + str(layer + 1)
            self.w.append(weight)
            self.b.append(bias)
            params[self.w[layer]] = np.random.normal(size=(self.layer_sizes[layer], self.layer_sizes[layer-1]))
            #params[self.b[layer]] = np.zeros((self.layer_sizes[layer], 1))
            params[self.b[layer]] = np.random.normal(size=(self.layer_sizes[layer], 1))

        #Initialize output layer, which depends on the number of classes being classified.
        weight = 'W' + str(self.num_layers + 1)
        bias = 'b' + str(self.num_layers + 1)
        self.w.append(weight)
        self.b.append(bias)
        params[weight] = np.random.normal(size=(self.num_classes, self.layer_sizes[self.num_layers-1]))
        #params[bias] = np.zeros((self.num_classes, 1))
        params[bias] = np.random.normal(size=(self.num_classes, 1))

        return params

    #Defines the training and dev (validation) datasets. The data and labels follow the standard
    #convention of each row being it's own training example, but my particular implementation
    #is optimized for columns being examples so I transpose the input data.
    def input_data(self, trainData, trainLabels, devData, devLabels):
        self.trainData = trainData.T
        self.trainLabels = (trainLabels.T).astype(int)
        self.devData = devData.T
        self.devLabels = (devLabels.T).astype(int)

    #Computes the softmax function for each element of a matrix, where the columns are
    #individual training examples. Uses shift invariance of the softmax function to avoid underflow.
    def softmax(self, x):
        maxs = np.amax(x, axis=0)
        s = np.exp(x - maxs) #softmax is shift invariant

        marginal = np.sum(s, axis=0)
        s /= marginal
        return s

    #Basic activation function. Supports sigmoid and relu activations
    def activation(self, x):
        if self.active_type == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
        else:
            s = np.maximum(0,x)
        return s

    #Calculates the cross entropy for a matrix of outputs and labels, where columns
    #represent training examples. Returns the average loss over all examples.
    def cross_entropy(self, output, labels):
        vector_output = (np.log(output)).flatten('F')
        vector_labels = labels.flatten('F')
        total_loss = -np.dot(vector_labels, vector_output) / output.shape[1]
        return total_loss

    #Calculates the Weston and Watkins SVM/hinge loss for a matrix of outputs and labels,
    #where columns represent examples. Returns average loss.
    def hinge(self, output, labels):
        true_pred = np.einsum('ij,ij->j', output, labels)
        raw = output - true_pred

        #The mimimum margin (Delta) is set to 1 without loss of generality, as it controls the
        #same thing as L2 regularization
        losses = np.maximum(0,raw + 1)
        total_loss = np.sum(losses) / output.shape[1]
        return total_loss

    #Forward pass of the neural net, also caches the layer outputs for backprop.
    #Returns the predictions and the cost, along with the cached values.
    def forward_prop(self, data, labels, params):
        cache = {}

        #Input layer
        z = np.matmul(params['W1'], data) + params['b1']
        cache['1'] = self.activation(z)

        #Hidden layers
        for layer in range(1, self.num_layers):
            z = np.matmul(params[self.w[layer]], cache[str(layer)]) + params[self.b[layer]]
            cache[str(layer + 1)] = self.activation(z)

        #Output layer; no activation function here
        z = np.matmul(params[self.w[self.num_layers]], cache[str(self.num_layers)]) + params[self.b[self.num_layers]]
        if self.loss == 'cross entropy':
            output = self.softmax(z)
            cost = self.cross_entropy(output, labels)
        else:
            output = z
            cost = self.hinge(output, labels)

        return cache, output, cost

    #Gradient of softmax cross entropy.
    def grad_cross_entropy(self, labels, activations):
        corrected_terms = activations - labels
        return corrected_terms / labels.shape[1]

    #Gradient of Weston-Watkins SVM/hinge loss.
    def grad_hinge(self, labels, activations):
        indicator = (activations > 0)
        gradient = indicator * (1 - labels)
        gradient[labels.astype(bool)] = -np.sum(gradient, axis=0) #make sure this indexing is valid
        return gradient

    #Backwards pass. Calculates gradients of all layers and biases using backpropagation.
    def backward_prop(self, data, cache, labels, params):
        output = params['o']
        deltas = {}

        #Deltas are gradients with respect to layer outputs; it is easy to calculate
        #these using backpropagation and also easy to go from the gradient of the layer to the
        #gradient of the weights and biases.
        if self.loss == 'cross entropy':
            deltas[str(self.num_layers + 1)] = self.grad_cross_entropy(labels, output)
        else:
            deltas[str(self.num_layers + 1)] = self.grad_hinge(labels, output)

        if self.active_type == 'sigmoid':
            for layer in range(self.num_layers, 0, -1):
                grad_sigmoid = (cache[str(layer)] * (1 - cache[str(layer)]))
                deltas[str(layer)] = np.matmul(params[self.w[layer]].T, deltas[str(layer + 1)]) * grad_sigmoid
        else:
            for layer in range(self.num_layers, 0, -1):
                grad_relu = (cache[str(layer)] != 0)
                deltas[str(layer)] = np.matmul(params[self.w[layer]].T, deltas[str(layer + 1)]) * grad_relu


        #From the calculated deltas, we can now easily calculate gradients of weights and biases
        #via backprop again.
        grads = {}
        grads['W1'] = np.matmul(deltas['1'], data.T)
        grads['b1'] = np.sum(deltas['1'], axis=1, keepdims=True)
        for layer in range(1, self.num_layers + 1):
            grads[self.w[layer]] = np.matmul(deltas[str(layer + 1)], cache[str(layer)].T)
            grads[self.b[layer]] = np.sum(deltas[str(layer + 1)], axis=1, keepdims=True)

        return grads

    '''
    Trains the neural network. A lot of hyperparameters, but it made the most sense to put them here.
    Training is done via minibatch gradient descent, the hyperparameters are taken as inputs to this function.
    Note that minibatches are taken sequentially, so the training data must be randomized prior to training.
    The learning rate is reduced by a factor of the decay rate every epoch. The dev set is used for validation 
    and has no effect on the training. If verbose is set to true, the progress is reported every epoch and a 
    plot of the performance history is generated at the end. Returns the set of trained parameters.
    '''
    def nn_train(self, reg_strength, epochs, batch_size, learning_rate, decay_rate, verbose):
        #Initialize variables, preprocess data.
        m = self.trainData.shape[1]
        training_losses = []
        dev_losses = []
        training_accs = []
        dev_accs = []
        epoch = 0
        iterations = m // batch_size * epochs

        #Initialize architecture
        params = self.init_params()

        #Train model.
        for it in range(iterations):

            #Select minibatches sequentially.
            start = (it*batch_size) % m
            end = start + batch_size
            data = self.trainData[:,start:end]
            labels = self.trainLabels[:,start:end]

            #Forward and backward passes, calculate gradients.
            cache, params['o'], cost = self.forward_prop(data, labels, params)
            grads = self.backward_prop(data, cache, labels, params)

            #Update parameters with both gradients and regularization.
            for weight in self.w:
                params[weight] -= learning_rate * (grads[weight] + 2 * reg_strength * params[weight])
            for bias in self.b:
                params[bias] -= learning_rate * grads[bias]

            #If verbose is set to true, provide progress reports every epoch.
            if it % (m//batch_size) == 0:
                epoch += 1
                learning_rate *= decay_rate
                if verbose:
                    #Evaluate model on both the entire training set and the entire dev set.
                    _, _, train_loss = self.forward_prop(self.trainData, self.trainLabels, params)
                    _, _, val_loss = self.forward_prop(self.devData, self.devLabels, params)
                    train_accuracy = self.nn_test(self.trainData, self.trainLabels, params)
                    dev_accuracy = self.nn_test(self.devData, self.devLabels, params)

                    print('Epoch %d, Train Accuracy %f, Dev Accuracy %f' % (epoch, train_accuracy, dev_accuracy))

                    #Store a history of losses and accuracies to plot at the end.
                    training_losses.append(train_loss)
                    dev_losses.append(val_loss)
                    training_accs.append(train_accuracy)
                    dev_accs.append(dev_accuracy)

        #If verbose is true, plot model performance history.
        if verbose:
            self.plot(training_accs, dev_accs, training_losses, dev_losses, epochs)
            plt.show()

        return params

    #Tests the neural network on a validation dataset and returns the accuracy.
    def nn_test(self, data, labels, params):
        _, output, _ = self.forward_prop(data, labels, params)
        accuracy = self.compute_accuracy(output, labels)
        return accuracy

    #Helper function to compute accuracies.
    def compute_accuracy(self, output, labels):
        accuracy = (np.argmax(output, axis=0) == np.argmax(labels, axis=0)).sum() * 1. / labels.shape[1]
        return accuracy

    #Plots the model performance history.
    def plot(self, training_accs, dev_accs, training_losses, dev_losses, epochs):
        plt.plot(range(1, epochs), training_accs[1:])
        plt.xlabel('# of Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Neural Network over Time')

        plt.plot(range(1, epochs), dev_accs[1:])
        plt.legend(('Training Set Accuracy', 'Dev Set Accuracy'))
        plt.figure()

        plt.plot(range(1, epochs), training_losses[1:])
        plt.xlabel('# of Epochs')
        plt.ylabel('Loss')
        plt.title('Loss of Neural Network over Time')

        plt.plot(range(1, epochs), dev_losses[1:])
        plt.legend(('Training Set Loss', 'Dev Set Loss'))