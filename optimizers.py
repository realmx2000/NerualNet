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


# Stochastic gradient descent: just follows the gradient down.
class SGD:
    # Just some book keeping for readability later on.
    def __init__(self, w, b):
        self.w = w
        self.b = b

    # No additional hyperparameters are needed for stochastic gradient descent.
    def verify_params(self, opt_params):
        return

    # Updates by simply taking a step opposite the gradient and adding L2 regularization.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        for weight in self.w:
            params[weight] -= learning_rate * (grads[weight] + 2 * reg_strength * params[weight])

        for bias in self.b:
            params[bias] -= learning_rate * grads[bias]


'''
Gradient Descent with Momentum: Treats parameters as an actual position on the 'landscape'
defined by the loss function. The gradient is then proportional the the acceleration,
and is integrated to get changes in the 'velocity' and the parameters themselves.
'''

class Momentum:
    # Store the velocity, since it will be updated later
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
        assert ('mu' in opt_params), 'Initial momentum must be specified with key \'mu\''
        assert ('anneal' in opt_params), 'Momentum annealing rate must be specified with key \'anneal\''
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


'''
Nesterov Accelerated Gradient: NAG improves on standard momentum optimization by noting
that we know a priori some information about where the parameters will end up on the next
step - we know that part of the update is the vector (momentum * velocity), and both of 
those quantities are known. We can therefore 'look ahead' at where we are going and evaluate
the gradient there instead of at our current position for a better estimate of the true 
acceleration we would feel in the real world. Note that in NAG, the parameters we store 
are the 'look-ahead' parameters, not the true parameters.

Among non-adaptive (and even many adaptive) approaches, NAG is considered the best optimizer, 
and is often considered as an alternative to the adam optimizer as a good default option.
'''
class NAG:
    # Store the velocity, since it will be updated later
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
        assert ('mu' in opt_params), 'Initial momentum must be specified with key \'mu\''
        assert ('anneal' in opt_params), 'Momentum annealing rate must be specified with key \'anneal\''

    # Updates parameters by integrating the acceleration (gradient) to update the velocity,
    # correcting the 'look-ahead' parameters back to the true parameters, and then making
    # both the new parameter update and the new look-ahead update using the new velocity.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        # Anneal the momentum based on the epoch
        momentum = 1 - ((1 - opt_params['mu']) * (opt_params['anneal'] ** opt_params['epoch']))

        for weight in self.w:
            # Store the current velocity before updating to the future velocity
            curr_velocity = self.velocities[weight]

            # Integrate the gradient (with L2 regularization) to get the change
            # in velocity, then update the velocity
            gradient_update = learning_rate * (grads[weight] + 2 * reg_strength * params[weight])
            self.velocities[weight] = momentum * self.velocities[weight] - gradient_update

            # Subtract off the current velocity update to get the actual position,
            # then update parameters with the new velocity, and finally do the
            # look-ahead step forwards
            params[weight] += (1 + momentum) * self.velocities[weight] - momentum * curr_velocity

        for bias in self.b:
            # Store the current velocity before updating to the future velocity
            curr_velocity = self.velocities[bias]

            # Integrate gradient to get change in velocity, then update velocity
            gradient_update = learning_rate * grads[bias]
            self.velocities[bias] = momentum * self.velocities[bias] - gradient_update

            # Integrate velocity to update parameters
            params[bias] += (1 + momentum) * self.velocities[bias] - momentum * curr_velocity

    # Corrects all the 'look-ahead' parameters back to their true values or the true
    # values back to the look-ahead parameters, depending on the direction argument.
    # (1 changes to true values, -1 changes to look-ahead values)
    def correct(self, params, opt_params, direction):
        for param in itertools.chain(self.w, self.b):
            params[param] -= direction * opt_params['mu'] * self.velocities[param]


######## Optimizers below here are adaptive per-parameter methods. ########

'''
Adaptive Gradient Descent: Dynamically adapts the learning rates of each parameter
by normalizing by the squared sum of previous gradients. This means that parameters
with high gradients end up with lower learning rates, and low gradients end up with
higher learning rates.
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


'''
Root Mean Square Propagation: A problem that comes up with adagrad is that the normalization 
factors monotonically increase, causing the learning rates to monotonically decrease. This 
aggressive reduction in learning often stops learning too early. RMSprop seeks to rectify this 
by using a moving average of previous squared gradients instead of the sum. Indeed, with this
approach the normalization factor is an estimate of the second moment, or the uncentered 
variance. If the gradient has high variance then intuitively we are unsure of which direction
to move in, so we should take a small step, and vice versa for low variance, which is realized
by the choice of normalization.
'''

class RMSprop:
    # Store the estimated variance, since it's updated as training progresses
    variance = {}

    # Sets a few variables for readability purposes and initializes the histories so that
    # the necessary keys are present.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.variance[param] = 0

    # RMSprop requires one additional hyperparameter, the historical decay rate ('decay').
    def verify_params(self, opt_params):
        assert ('decay' in opt_params), 'Average decay rate must be specified with key \'decay\''
        return

    # Updates parameters by first taking a convex combination of the history and the
    # new squared gradient, then updating the parameters with the normalized learning rate.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        decay = opt_params['decay']

        for weight in self.w:
            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            # Update the variance estimate by taking a convex combination of the previous history
            # and the new squared gradient
            self.variance[weight] = decay * self.variance[weight] + (1 - decay) * np.square(reg_gradient)

            # Update parameters with normalized learning rate
            params[weight] -= learning_rate * reg_gradient / (np.sqrt(self.variance[weight]) + 1e-8)

        for bias in self.b:
            # Take convex combination of previous history and new squared gradient
            self.variance[bias] = decay * self.variance[bias] + (1 - decay) * np.square(grads[bias])

            # Update parameters with normalized learning rate
            params[bias] -= learning_rate * grads[bias] / (np.sqrt(self.variance[bias]) + 1e-8)


'''
Adadelta: The adadelta algorithm makes a further refinement to the RMSprop algorithm by 
scaling the regularized gradients by a moving average of previous updates in the update
rule. Intuitively, if the gradient has pointed in a certain direction for a long time, the
parameters should probably keep moving in that direction. The adagrad scaling prevents 
sudden gradient changes (from saddle points or chance variations in the loss) from causing 
the algorithm to drastically change direction; the parameters have 'momentum', so to speak.
'''

class Adadelta:
    # Store the variance estimate and the average update for later updates
    variance = {}
    deltas = {}

    # Sets a few variables for readability purposes and initializes the estimates so that
    # the necessary keys are present.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.variance[param] = 0
            self.deltas[param] = 0

    # Adadelta requires one additional hyperparameter, the decay rate for the average estimators.
    def verify_params(self, opt_params):
        assert ('decay' in opt_params), 'Average decay rates must be specified with key \'decay\''
        return

    # Updates the parameters by updating the variance estimate and the average of updates,
    # then performing the update by scaling by the average update divided by the estimated
    # variance.
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        decay = opt_params['decay']

        for weight in self.w:
            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            # Update the variance estimate by taking a convex combination of the previous history
            # and the new squared gradient
            self.variance[weight] = decay * self.variance[weight] + (1 - decay) * np.square(reg_gradient)

            # Compute the update using the normalized learning rate, and add it into the moving
            # average of deltas
            delta = learning_rate * reg_gradient / (np.sqrt(self.variance[weight]) + 1e-8)
            self.deltas[weight] = decay * self.deltas[weight] + (1 - decay) * np.square(delta)

            # Update the parameters, scaling the gradient by the average recent update divided
            # by the estimated variance
            params[weight] -= np.sqrt(self.deltas[weight] / (self.variance[weight] + 1e-16)) * reg_gradient

        for bias in self.b:
            # Take convex combination of previous history and new squared gradient
            self.variance[bias] = decay * self.variance[bias] + (1 - decay) * np.square(grads[bias])

            # Compute update and add to moving average
            delta = learning_rate * grads[bias] / (np.sqrt(self.variance[bias]) + 1e-8)
            self.deltas[bias] = decay * self.deltas[bias] + (1 - decay) * np.square(delta)

            # Update parameters with proper scaling
            params[bias] -= np.sqrt(self.deltas[bias] / (self.variance[bias] + 1e-16)) * grads[bias]


'''
Adaptive Moment Estimation: The Adam optimizer makes two significant changes to the RMSprop 
algorithm. First, the gradient in the update rule is replaced by a moving average of previous 
gradients. This is to reduce noise; unless the batch size is close to the entire dataset the
minibatch gradient does not in general match the true gradient. By taking an average of gradients,
we get a better estimate for the true gradient (and also avoid the sudden change problem that 
Adagrad sought to rectify. The second change is that both the estimate for the first moment 
(the mean of the gradient) and the second moment (the uncentered variance of the gradient) are 
divided by an additional term, called the 'bias correction' term. This is because the zero
initialization of the estimators makes them biased; these correction terms fix that problem.

Generally, the Adam optimizer is considered to be the best optimizer and is a good default choice,
along with NAG.
'''

class Adam:
    # Store the biased estimates for the first and second moments so they can be continuously updated
    mean = {}
    variance = {}

    # Sets a few variables for readability purposes and initializes the estimates so that
    # the necessary keys are present.
    def __init__(self, w, b):
        self.w = w
        self.b = b

        for param in itertools.chain(w, b):
            self.mean[param] = 0
            self.variance[param] = 0

    # The adam optimizer requires two additional hyperparameters: the decay rate for the moving average
    # in the mean estimator, and the decay rate for the variance estimator.
    def verify_params(self, opt_params):
        assert ('decay1' in opt_params), 'First order moment decay must be specified with key \'decay1\''
        assert ('decay2' in opt_params), 'Second order moment decay must be specified with key \'decay2\''
        return

    # Updates parameters by first updating the biased estimates for the moments, then adding
    # bias correction terms, then performing the update with the mean as the gradient
    # and normalizing with the variance
    def update(self, params, learning_rate, reg_strength, grads, opt_params):
        b1 = opt_params['decay1']
        b2 = opt_params['decay2']
        t = opt_params['iteration']

        for weight in self.w:
            # Regularize the gradient with L2 regularization
            reg_gradient = grads[weight] + 2 * reg_strength * params[weight]

            # Update the biased mean estimate, then calculate the unbiased estimate
            self.mean[weight] = b1 * self.mean[weight] + (1 - b1) * reg_gradient
            m_hat = self.mean[weight] / (1 - (b1 ** t))

            # Update the biased variance estimate, the calculate the unbiased estimate
            self.variance[weight] = b2 * self.variance[weight] + (1 - b2) * np.square(reg_gradient)
            v_hat = self.variance[weight] / (1 - (b2 ** t))

            # Update parameters using the mean as the gradient and normalizing
            # by the variance
            params[weight] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        for bias in self.b:
            # Update the biased mean estimate, then calculate the unbiased estimate
            self.mean[bias] = b1 * self.mean[bias] + (1 - b1) * grads[bias]
            m_hat = self.mean[bias] / (1 - (b1 ** t))

            # Update the biased variance estimate, the calculate the unbiased estimate
            self.variance[bias] = b2 * self.variance[bias] + (1 - b2) * np.square(grads[bias])
            v_hat = self.variance[bias] / (1 - (b2 ** t))

            # Update parameters using the mean as the gradient and normalizing
            # by the variance
            params[bias] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
