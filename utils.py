def softmax(a):
    e_a = np.exp(x - np.max(x))
    return e_a / e_a.sum(axis=0)

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
