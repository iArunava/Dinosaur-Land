import numpy
from utils import *

class RNN:
    def rnn_forward(self, X, Y, a0, parameters):
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
            a[t], y_hat[t], cache = self.rnn_cell_forward(x[t], a[t-1], parameters)

            # Get the loss
            loss -= np.log(y_hat[t][Y[t], 0])

            # Save the value of new a_next
            a[t] = a_next

        # Store values needed for backpropagation
        caches = (y_hat, a, x)

        return loss, cache

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

        return a_next, yt_pred, cache

    def rnn_cell_backward(self, da_next, cache):
        """
        Implements the backward pass for a single RNN cell
        """

        # Retrieve the values from the cache
        a_next, a_prev, xt, parameters = cache

        # Retrieve values from parameters
        Wax = parameters['Wax']
        Waa = parameters['Waa']
        Wya = parameters['Wya']
        ba = parameters['ba']
        by = parameters['by']

        # Compute the gradient of tanh with respect to a_next
        dtanh = (1 - a_next ** 2) * da_next

        # Compute the gradient of the loss with respect to Wax
        dxt = np.dot(Wax.T, dtanh)
        dWax = np.dot(dtanh, xt.T)

        # Compute the gradient with respect to Waa
        da_prev = np.dot(Waa.T, dtanh)
        dWaa = np.dot(dtanh, a_prev.T)

        # Compute the gradients with respect to b
        dba = np.sum(dtanh, axis=1, keepdims=True)

        # Store the gradients
        gradients = {'dxt' : dxt, 'da_prev' : da_prev,
                     'dWax' : dWax, 'dWaa' : dWaa, 'dba' : dba}

        return gradients

    def rnn_backward(self, da, caches):
        """
        Implement the backward pass for a RNN over an entire sequence of input data
        """

        # Retrieve values for the first cache to get dimension info
        (caches, x) = caches
        (a1, a0, x, parameters) = caches[0]

        # Retrieve dim info app from da and x1
        n_a, m, T_x = da.shape
        n_x, m = x1.shape

        # Initialize the gradients with right sizes
        dx = np.zeros((n_x, m, T_x))
        dWax = np.zeros((n_a, n_x))
        dWaa = np.zeros((n_a, n_a))
        dba = np.zeros((n_a, 1))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))

        # Loop through the timesteps reversed
        for t in reversed(range(T_x)):
            # Compute the gradients at the time step t
            gradients = self.rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

            # Retrieve derivatives from gradients
            dxt, = gradients['dxt']
            da_prevt = gradients['da_prev']
            dWaxt = gradients['dWax']
            dWaat = gradients['dWaa']
            dbat = gradients['dba']

            # Store the gradient with respect to the input for this timestep
            dx[:, :, t] = dxt

            # Increment the global derivatives
            dWax += dWaxt
            dWaa += dWaat
            dba += dbat

        # Setting da0 to the gradient which has been propagated back through time
        da0 = da_prevt

        # Store the gradients in dictionary
        gradients = {'dx': dx, 'da0': da0, 'dWax': dWax, 'dWaa': dWaa, 'dba': dba}

        return gradients

    def update_parameters(self, parameters, gradients, lr):
        parameters['Wax'] += -lr * gradients['dWax']
        parameters['Waa'] += -lr * gradients['dWaa']
        parameters['Wya'] += -lr * gradients['dWya']
        parameters['ba'] += -lr * gradients['dba']
        parameters['Wby'] += -lr * gradients['dby']
        return parameters

    def optimize(self, X, Y, a_prev, parameters, learning_rate=0.01):
        """
        Execute one step of the optimization to train the model
        """

        # Forward propagate
        loss, cache, _, _ = self.rnn_forward(X, Y, a_prev, parameters)

        # Backpropagate through time
        gradients, a = self.rnn_backward(X, Y, parameters, cache)

        # Clip gradients
        gradients = clip(gradients, 5)

        # Update parameters
        parameters = self.update_parameters(parameters, gradients, learning_rate)

        return loss, gradients, a[len(X) - 1]

    def initialize_parameters(n_a, n_x, n_y):
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
