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
