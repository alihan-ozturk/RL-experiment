import numpy as np

def argmax(q):
    top = float("-inf")
    ties = []
    for i in range(len(q)):
        if q[i] == top:
            ties.append(i)
        elif q[i] > top:
            top = q[i]
            ties = [i]
    return np.random.choice(ties)