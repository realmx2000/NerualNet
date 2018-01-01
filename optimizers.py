'''
 Created by Zhaoyu Lou on 12/31/17.
'''

import numpy as np
import itertools

'''
This file supplies the optimizers for the neural network. Each optimizer class
has a verify_params method which checks that all necessary additional hyperparameters 
are in the opt_params dictionary with the proper keys, and an update method which
updates the network parameters.
'''

# Stochastic gradient descent; just follows the gradient down.
class SGD:

    # Just some book keeping for readability later on.
    def __init__(self, w, b):
        self.w = w
        self.b = b

    # No additional parameters are needed for stochastic gradient descent.
    def verify_params(self, opt_params):
        return

    # Updates by simply taking a step opposite the gradient and adding L2 regularization.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        for weight in self.w:
            params[weight] -= learning_rate * (grads[weight] + 2 * reg_strength * params[weight])

        for bias in self.b:
            params[bias] -= learning_rate * grads[bias]

        return params

'''
Momentum optimization. Treats parameters as an actual position on the 'landscape'
defined by the loss function. The gradient is then proportional the the acceleration,
and is integrated to get changes in the 'velocity' and the parameters themselves.
'''
class Momentum:
    # Store the velocity, since the gradient changes the velocity, not the parameters
    velocities = {}

    # Initializes the velocity dictionary so that all the required keys are present.
    # Also sets a few variables for readability purposes.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.velocities[param] = 0

    # Momentum optimization requires 2 additional hyperparameters: the initial
    # momentum (mu) and the rate at which the momentum is annealed towards 1 (anneal).
    def verify_params(self, opt_params):
        #assert(opt_params.contains('mu')), 'Initial momentum must be specified with key \'mu\''
        #ssert(opt_params.contains('anneal')), 'Momentum annealiing rate must be specified with key \'anneal\''
        return

    # Updates parameters by first integrating the acceleration (the gradient) to update the velocity,
    # then integrating the velocity to update the position (parameters). Returns the updated parameters.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        # Anneal the momentum based on the epoch
        momentum = 1 - ((1 - opt_params['mu']) * (opt_params['anneal'] ** opt_params['epoch']))

        for weight in self.w:

            # Integrate the gradient (with L2 regularization) to get the change
            # in velocity, then update the velocity
            gradient_update = learning_rate * (grads[weight] + 2 * reg_strength * params[weight])
            self.velocities[weight] = momentum * self.velocities[weight] - gradient_update

            # Integrate the velocity to update the parameters
            params[weight] += self.velocities[weight]

        for bias in self.b:

            # Integrate gradient to get change in velocity, then update velocity
            gradient_update = learning_rate * grads[bias]
            self.velocities[bias] = momentum * self.velocities[bias] - gradient_update

            # Integrate velocity to update parameters
            params[bias] += self.velocities[bias]

        return params

# TODO: Figure out updates for Nesterov
#class Nesterov:

'''
Adaptive gradient descent. Dynamically adapts the learning rates of each parameter
by normalizing by the squared sum of previous gradients. This means that parameters
with high gradients have their learning rates reduced, and low gradients have their
learning rates increased.
'''
class Adagrad:
    # Store the sum, since it's updated as training progresses
    histories = {}

    # Sets a few variables for readability purposes and initializes the histories
    # so that all the necessary keys are present.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.histories[param] = 0

    # No additional hyperparameters are needed for adagrad.
    def verify_params(self, opt_params):
        return

    # Updates parameters by adding the square of the current gradient to the history,
    # then stepping down the gradient with the normalized learning rate.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):

        for weight in self.w:

            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            # Update the history with the squared regularized gradient
            self.histories[weight] += np.square(reg_gradient)
            params[weight] -= learning_rate * reg_gradient / (np.sqrt(self.histories[weight]) + 1e-8)

        for bias in self.b:

            # Update the history with the squared gradient
            self.histories[bias] += np.square(grads[bias])

            # Update the parameters using the normalized learning rate
            params[bias] -= learning_rate * grads[bias] / (np.sqrt(self.histories[bias]) + 1e-8)

        return params

'''
Root mean square propagation: A problem that comes up with adagrad is that the normalization 
factors monotonically increase, causing the learning rates to monotonically decrease. This 
aggressive normalization often stops learning too early. RMSprop seeks to rectify this by 
using a moving average of previous squared gradients instead of the sum.
'''
class RMSprop:
    # Store the moving average, since it's updated as training progresses
    histories = {}

    # Sets a few variables for readability purposes and initializes the histories so that
    # the necessary keys are present.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.histories[param] = 0

    # RMSprop requires one additional hyper parameter, the historical decay rate ('decay').
    def verify_params(self, opt_params):
        #assert(opt_params.contains('decay'), 'historical decay rate must be specified with key \'decay\''
        return

    # Updates parameters by first taking a convex combination of the history and the
    # new squared gradient, then updating the parameters with the normalized learning rate.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        decay = opt_params['decay']

        for weight in self.w:

            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            # Decay the history by taking a convex combination of the previous history
            # and the new squared gradient
            self.histories[weight] = decay * self.histories[weight] + (1 - decay) * np.square(reg_gradient)

            params[weight] -= learning_rate * reg_gradient / (np.sqrt(self.histories[weight]) + 1e-8)

        for bias in self.b:

            # Take convex combination of previous history and new squared gradient
            self.histories[bias] = decay * self.histories[bias] + (1 - decay) * np.square(grads[bias])

            # Update parameters with normalized learning rate
            params[bias] -= learning_rate * grads[bias] / (np.sqrt(self.histories[bias]) + 1e-8)

        return params

class Adam:
    m = {}
    v = {}
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.m[param] = 0
            self.v[param] = 0

    def verify_params(self, opt_params):
        #assert(opt_params.contains('decay1')), 'First order moment decay must be specified with key \'decay\''
        #assert(opt_params.contains('decay2')), 'Second order moment decay must be specified with key \'decay\''
        return

    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        b1 = opt_params['decay1']
        b2 = opt_params['decay2']
        t = opt_params['iteration']

        for weight in self.w:
            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            self.m[weight] = b1 * self.m[weight] + (1 - b1) * reg_gradient
            m_hat = self.m[weight] / (1 - (b1 ** t))

            self.v[weight] = b2 * self.v[weight] + (1 - b2) * np.square(reg_gradient)
            v_hat = self.v[weight] / (1 - (b2 ** t))

            params[weight] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        for bias in self.b:
            self.m[bias] = b1 * self.m[bias] + (1 - b1) * grads[bias]
            m_hat = self.m[bias] / (1 - (b1 ** t))

            self.v[bias] = b2 * self.v[bias] + (1 - b2) * np.square(grads[bias])
            v_hat = self.v[bias] / (1 - (b2 ** t))

            params[bias] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        return params
