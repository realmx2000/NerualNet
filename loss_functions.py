import numpy as np

'''
This file supplies the loss functions for the neural network. Each loss class has a calculate_loss
method which calculates the loss given the outputs from the last layer (logits) for forwards 
propagation, and a grad_loss method which calculates the gradient of the loss function for 
backpropagation.
'''

#Softmax cross entropy loss
class CrossEntropy:

    #Computes the softmax function for each element of a matrix, where the columns are
    #individual training examples. Uses shift invariance of the softmax function to avoid underflow.
    def softmax(self, x):
        maxs = np.amax(x, axis=0)
        s = np.exp(x - maxs) #softmax is shift invariant

        marginal = np.sum(s, axis=0)
        s /= marginal
        return s

    #Calculates the cross entropy for a matrix of outputs and labels, where columns
    #represent training examples. Returns the average loss over all examples.
    def calculate_loss(self, logits, labels):
        output = self.softmax(logits)
        vector_output = (np.log(output)).flatten('F')
        vector_labels = labels.flatten('F')
        total_loss = -np.dot(vector_labels, vector_output) / output.shape[1]
        return total_loss, output

    #Gradient of softmax cross entropy.
    def grad_loss(self, labels, activations):
        corrected_terms = activations - labels
        return corrected_terms / labels.shape[1]

#Weston-Watkins hinge/support vector machine loss
class Hinge:

    #Calculates the Weston and Watkins SVM/hinge loss for a matrix of outputs and labels,
    #where columns represent examples. Returns average loss.
    def calculate_loss(self, logits, labels):
        true_pred = np.einsum('ij,ij->j', logits, labels)
        raw = logits - true_pred

        #The mimimum margin (Delta) is set to 1 without loss of generality, as it controls the
        #same constraint as L2 regularization
        losses = np.maximum(0,raw + 1)
        total_loss = np.sum(losses) / logits.shape[1]
        return total_loss, logits

    #Gradient of Weston-Watkins SVM/hinge loss.
    def grad_loss(self, labels, activations):
        indicator = (activations > 0)
        gradient = indicator * (1 - labels)
        gradient[labels.astype(bool)] = -np.sum(gradient, axis=0)
        return gradient