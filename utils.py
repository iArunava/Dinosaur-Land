def softmax(a):
    e_a = np.exp(x - np.max(x))
    return e_a / e_a.sum(axis=0)
