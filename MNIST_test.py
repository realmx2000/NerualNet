'''
 Created by Zhaoyu Lou on 12/24/17.
'''

import pickle
import numpy as np

from neuralNet import NeuralNet


# Load images and labels.
def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


# Generates the one-hot encoded vector labels
def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


# Load MNIST dataset and train neural network on it.
def main():
    np.random.seed(100)
    # Load training dataset and randomize.
    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)

    # Create train-dev split.
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]
    devData = trainData[0:10000, :]
    devLabels = trainLabels[0:10000, :]
    trainData = trainData[10000:, :]
    trainLabels = trainLabels[10000:, :]

    # Center and whiten data.
    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    # Load test dataset.
    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    # Create model and train.
    net = NeuralNet(2, [300, 150], trainData.shape[1], 10, 'relu', 'cross entropy', 'rmsprop')
    print('Beginning Training.')
    net.input_data(trainData, trainLabels, devData, devLabels)

    # Since we are using SGD, no further parameters are needed for the optimizer
    optimizer_params = {'mu': 0.5, 'anneal': 0.95, 'decay' : 0.9, 'b1': 0.9, 'b2': 0.999}
    params = net.nn_train(0.0001, 50, 1000, 0.001, 0.99, True, optimizer_params)

    # params = net.nn_train(0.0001, 50, 1000, 7.5, 0.98, True, optimizer_params)

    # Save parameters.
    pickle.dump(params, open('params.pickle', 'wb'))

    # Test model on test dataset and report accuracy.
    accuracy = net.nn_test(testData.T, testLabels.T, params)
    print('Test accuracy: %f' % accuracy)


if __name__ == '__main__':
    main()
