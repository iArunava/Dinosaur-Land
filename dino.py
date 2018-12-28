import numpy as np
from utils import *
from RNN import *

rnn = RNN()

# Read the dataset
data = open('./dataset/dinosaurs.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('There are %d total characters and %d unique characters in your data' % (data_size, vocab_size))

# Create the int2char and char2int dictionaries
char2int = {ch : i for i, ch in enumerate(chars)}
int2char = {i : ch for i, ch in enumerate(chars)}
print ('Integer to Character Mapping: ')
print (int2char)

# Retrieve n_x, n_y from vocab_size
n_x, n_y = vocab_size, vocab_size

# Initialize parameters
parameters = initialize_parameters(vocab_size, dino_names)
seq_length = 7

# Initialize Loss
loss = get_initial_loss(vocab_size, seq_length)

# Build list of all dinosaurs
with open('dataset/dinosaurs.txt', 'r') as f:
    dinos = f.readlines()
dinos = [d.lower().strip() for dino in dinos]

# Initialize the hidden state
a_prev = np.zeros((n_a, 1))

# Optimization Loop
for i in range(4):
    # Create a training example
    idx = i % len(dinos)
    X = [None] + [char2int[ch] for ch in dinos[idx]]
    Y = X[1:] + [char2int['\n']]

    # Perform one optimization step
    curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, 0.01)

    # Latency trick to keep the loss smooth
    loss = smooth(loss, curr_loss)

    # Every n iterations generate samples
    if j % 2000 == 0:
        print ('Iteration: %d, Loss: %f' % (i, loss) + '\n')

        # Sample dino names
        for name in range(dino_names):
            sampled_indices = sample(parameters, char2int)
            print_sample(sampled_indices, int2char)

        print ('\n')
