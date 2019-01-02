import numpy as np
from utils import *
from RNN import *
import argparse

def initiate_training(FLAGS, data_size, vocab_size, char2int, int2char):


    # Initialize RNN
    n_a = FLAGS.rnn_hidden_units # Number of RNN units
    rnn = RNN(n_a, vocab_size, vocab_size, FLAGS.learning_rate)

    # Sample Length
    sample_length = FLAGS.sample_length

    # Initialize Loss
    loss = 0
    #loss = get_initial_loss(vocab_size, sample_length)

    # Build list of all dinosaurs
    with open('dataset/dinosaurs.txt', 'r') as f:
        dinos = f.readlines()
    dinos = [dino.lower().strip() for dino in dinos]

    # Initialize the hidden state
    a_prev = np.zeros((n_a, 1))

    # Optimization Loop
    for i in range(1, FLAGS.epochs):
        # Create a training example
        idx = i % len(dinos)
        X = [None] + [char2int[ch] for ch in dinos[idx]]
        Y = X[1:] + [char2int['\n']]

        # Perform one optimization step
        curr_loss, gradients, a_prev = rnn.optimize(X, Y, a_prev)

        # Latency trick to keep the loss smooth
        loss = smooth_loss(loss, curr_loss)

        # Every n iterations generate samples
        if i % 2000 == 0:
            print ('Iteration: %d, Loss: %f' % (i, loss) + '\n')

            # Sample dino names
            parameters = rnn.get_weights()
            for name in range(sample_length):
                sampled_indices = sample(parameters, char2int)
                print_sample(sampled_indices, int2char)

            print ('\n')

    # Save the weights
    rnn.save_weights(FLAGS.saved_model_extension)

    print ('[INFO]Training Complete!!')

def get_new_dinos(FLAGS, data_size, vocab_size, char2int, int2char):
    # Initialize rnn
    rnn = RNN(FLAGS.rnn_hidden_units, vocab_size, vocab_size, FLAGS.learning_rate)

    # Load the weights
    rnn.load_model(FLAGS.saved_model_extension)

    # Start sampling
    print ()
    for ii in range(FLAGS.sample_length):
        print_sample(sample(rnn.get_weights(), char2int), int2char)
    print ()
    
    print ('[INFO]Sampling Complete!! \n How do you like the new dinosaur names?')
