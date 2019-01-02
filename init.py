import numpy as np
import argparse
import os
from dino import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs',
    type=int,
    default=10000,
    help='The number of epochs')

parser.add_argument('-lr', '--learning-rate',
    type=float,
    default=0.01,
    help='The learning rate')

parser.add_argument('-sme', '--saved-model-extension',
    type=str,
    default='1',
    help='The extension in the file name with which the model is saved')

parser.add_argument('-sl', '--sample-length',
    type=int,
    default=5,
    help='The number of dino names to sample')

parser.add_argument('-t', '--train',
    type=int,
    default=1,
    help='Whether to train or test. 1 = Train | others = Test')

parser.add_argument('-d', '--dataset',
    type=str,
    default='./dataset/dinosaurs.txt',
    help='The dataset location')

parser.add_argument('-rhu', '--rnn-hidden-units',
    type=int,
    default=50,
    help='The number of hidden units in RNN')

parser.add_argument('-md', '--model-dir',
    type=str,
    default='./model/',
    help='The directory where the model is stored.')

parser.add_argument('-s', '--save-model',
    type=int,
    default=1,
    help='Save the model. If 1 else 0. Default 1')

FLAGS, unparsed = parser.parse_known_args()

# Read the dataset
data = open(FLAGS.dataset, 'r').read()
data = data.lower()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print ('There are %d total characters and %d unique characters in your data' % (data_size, vocab_size))

# Create the int2char and char2int dictionaries
char2int = {ch : i for i, ch in enumerate(chars)}
int2char = {i : ch for i, ch in enumerate(chars)}
print ('Integer to Character Mapping: ')
print (int2char)

print (FLAGS.train)
if (FLAGS.train):
    print ('[INFO]Intiating Model Training...')
    initiate_training(FLAGS, data_size, vocab_size, char2int, int2char)
else:
    print ('[INFO]Getting New Dinos using saved models')
    get_new_dinos(FLAGS, data_size, vocab_size, char2int, int2char)

print ('[INFO]Exiting...')
