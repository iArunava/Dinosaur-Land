def softmax(a):
    e_a = np.exp(x - np.max(x))
    return e_a / e_a.sum(axis=0)

def smooth_loss(loss, curr_loss):
    return loss * 0.999 + curr_loss * 0.001

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length

def print_sample(sampled_indices, int2char):
    dino = ''.join(int2char[idx] for idx in sampled_indices)
    print (dino[0].upper() + dino[1:])
    
def clip(gradients, max_val):
    dWaa = gradients['dWaa']
    dWax = gradients['dWax']
    dWya = gradients['dWya']
    dba = gradients['dba']
    dby = gradients['dby']

    # Clip gradients b/w -max_val and max_val
    for gradient in [dWax, dWaa, dWya, dba, dby]:
        np.clip(gradient, -max_val, max_val, out=gradient)

    # Store the updated gradients
    gradients = {'dWaa' : dWaa,
                 'dWax' : dWax,
                 'dWya' : dWya,
                 'dba'  : dba,
                 'dby'  : dby}

    return gradients

# Function to sample from the model
def sample(parameters, char2int):
    """
    Sample a sequence of characters according to a sequence of probability
    distributions output of the RNN
    """

    # Retrieve parameters and relevant shapes from parameters
    Waa = parameters['Waa']
    Wax = parameters['Wax']
    Wya = parameters['Wya']
    Wba = parameters['Wba']
    Wby = parameters['Wby']

    # Get vocab_size
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # Create a one-hot vector x for the first character
    x = np.zeros((vocab_size, 1))

    # Initialize a_prev as zeros
    a_prev = np.zeros((n_a, 1))

    # Create a list which will keep the sampled indices
    indices = []

    # idx is a flag to detect a newline character
    idx = -1

    # Set the counter to initial value
    # This is the value to check the number of sampled characters
    counter = 0

    # Getting the idx of the newline char
    newline_char = char2int['\n']

    # Loop over timesteps and sample at each timestep
    while (idx != newline_char and counter != 50):
        # Forward Prop
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)

        # Getting the idx with probability 'y'
        idx = np.random.choice(np.arange(vocab_size), p=y.ravel())

        # Append to the indices
        indices.append(idx)

        # Overwrite the input character
        x = np.zeros((vocab_size, 1))
        x[idx] = 1

        # Update 'a_prev' to be 'a'
        a_prev = a

    # If we have sampled for 50 times then adding a newline character
    if counter == 50:
        indices.append(char2int['\n'])

    return indices
