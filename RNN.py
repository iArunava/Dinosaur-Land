import numpy
from utils import *

def rnn_cell_forward(xt, a_prev, parameters):
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
