import numpy
from utils import *

class RNN:
    def rnn_forward(self, x, a0, parameters):
        """
        Implements the complete forward propagation in RNN
        """
        # Initialize caches which will contain all the caches required for backpropagation
        caches = []

        # Retrieve the dimensions from shapes of x and parameters['Wya']
        n_x, m, T_x = x.shape
        n_y, n_a = parameters['Wya']

        # Initialize variables with zeros that will store the cell states and
        # predictions for each cell
        a = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))

        # Initialize the init a
        a_next = a0

        # Loop over all the timesteps
        for t in range(T_x):
            # Update the next hidden state
            # Get the prediction
            # And the cache for the forward pass
            a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

            # Save the value of new a_next
            a[:, :, t] = a_next

            # Save the prediction
            y_pred[:, :, t] = yt_pred

            # Append to caches
            caches.append(cache)

        # Store values needed for backpropagation
        caches = (caches, x)

        return a, y_pred, caches

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

        # Store values you need for backpropagation
        cache = (a_next, a_prev, xt, parameters)

        return a_next, yt_pred, cache
