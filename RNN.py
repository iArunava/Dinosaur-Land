import numpy as np
from utils import *

class RNN:
    def rnn_forward(self, X, Y, a0, parameters, vocab_size=27):
        """
        Implements the complete forward propagation in RNN
        """

        # Initialize x, a and y_hat as empty dictionaries
        x, a, y_hat = {}, {}, {}

        # Copy the first cell state to the dictionary
        a[-1] = np.copy(a0)

        # Initialize your loss to 0
        loss = 0

        # Loop over all the timesteps
        for t in range(len(X)):

            # Set x[t] to be a one hot vector representing the t'th character in X
            x[t] = np.zeros((vocab_size, 1))
            if (X[t] != None):
                x[t][X[t]] = 1

            # Update the next hidden state
            # Get the prediction
            # And the cache for the forward pass
            a[t], y_hat[t] = self.rnn_cell_forward(x[t], a[t-1], parameters)

            # Get the loss
            loss -= np.log(y_hat[t][Y[t], 0])

        # Store values needed for backpropagation
        caches = (y_hat, a, x)

        return loss, caches

    def rnn_cell_forward(self, xt, a_prev, parameters):
        """
        Implements a single forward step in a RNN cell
        """

        # Retrieve the parameters
        Wax = parameters['Wax']
        Waa = parameters['Waa']
        Wya = parameters['Wya']
        ba = parameters['ba']
        by = parameters['by']

        # Compute the next cell state
        a_next = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, a_prev) + ba)

        # Compute the output of the current cell
        yt_pred = softmax(np.matmul(Wya, a_next) + by)

        return a_next, yt_pred

    def rnn_cell_backward(self, dy, gradients, parameters, x, a, a_prev):
        """
        Implements the backward pass for a single RNN cell
        """

        # Retrieve values from parameters
        Wax = parameters['Wax']
        Waa = parameters['Waa']
        Wya = parameters['Wya']
        ba = parameters['ba']
        by = parameters['by']

        #Calculate gradients
        gradients['dWya'] += np.dot(dy, a.T)
        gradients['dby'] += dy

        da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']
        daraw = (1 - a * a) * da

        gradients['dba'] += daraw
        gradients['dWax'] += np.dot(daraw, x.T)
        gradients['dWaa'] += np.dot(daraw, a_prev.T)
        gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)

        return gradients

    def rnn_backward(self, X, Y, parameters, cache):
        """
        Implement the backward pass for a RNN over an entire sequence of input data
        """

        # Initialize the gradients
        gradients = {}

        # Retrieve from cache
        (y_hat, a, x) = cache

        # Retrieve values from parameters
        Wax = parameters['Wax']
        Waa = parameters['Waa']
        Wya = parameters['Wya']
        ba = parameters['ba']
        by = parameters['by']

        # Initialize the gradients with right sizes
        gradients['dWax'] = np.zeros_like(Wax)
        gradients['dWaa'] = np.zeros_like(Waa)
        gradients['dWya'] = np.zeros_like(Wya)
        gradients['dba'] = np.zeros_like(ba)
        gradients['dby'] = np.zeros_like(by)
        gradients['da_next'] = np.zeros_like(a[0])

        # Loop through the timesteps reversed
        for t in reversed(range(len(X))):
            # Get a copy of the output in the current timestep
            dy = np.copy(y_hat[t])

            dy[Y[t]] -= 1

            # Compute the gradients at the time step t
            gradients = self.rnn_cell_backward(dy, gradients, parameters, x[t], a[t], a[t-1])

        return gradients, a

    def update_parameters(self, parameters, gradients, lr):
        parameters['Wax'] += -lr * gradients['dWax']
        parameters['Waa'] += -lr * gradients['dWaa']
        parameters['Wya'] += -lr * gradients['dWya']
        parameters['ba'] += -lr * gradients['dba']
        parameters['by'] += -lr * gradients['dby']
        return parameters

    def optimize(self, X, Y, a_prev, parameters, learning_rate=0.01):
        """
        Execute one step of the optimization to train the model
        """

        # Forward propagate
        loss, cache = self.rnn_forward(X, Y, a_prev, parameters)

        # Backpropagate through time
        gradients, a = self.rnn_backward(X, Y, parameters, cache)

        # Clip gradients
        gradients = clip(gradients, 5)

        # Update parameters
        parameters = self.update_parameters(parameters, gradients, learning_rate)

        return loss, gradients, a[len(X) - 1]

    def initialize_parameters(self, n_a, n_x, n_y):
        """
        Initialize parameters
        """

        Wax = np.random.randn(n_a, n_x) * 0.01
        Waa = np.random.randn(n_a, n_a) * 0.01
        Wya = np.random.randn(n_y, n_a) * 0.01
        ba = np.zeros((n_a, 1))
        by = np.zeros((n_y, 1))

        parameters = {'Wax' : Wax,
                      'Waa' : Waa,
                      'Wya' : Wya,
                      'ba'  : ba,
                      'by'  : by}

        return parameters
