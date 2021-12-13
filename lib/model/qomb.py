import numpy as np
from scipy.special import softmax


def update(Q, ops, rew, D=1):
    D = min(D, len(Q.shape))

    ops = list(set(ops))
    D1 = ops
    D2 = np.array([ops] * len(ops)).T
    Q[D1, D2] = np.maximum(Q[D1, D2], rew)

    return Q


def sample(Q, anchors, sandwich=False):
    D = len(Q)
    cumulative = np.zeros(D)
    for i, a in enumerate(anchors):
        row = anchors[a]
        if sandwich:
            row = softmax(row)
        cumulative += row

    prob_dist = softmax(cumulative)
    return np.random.choice(D, p=prob_dist)
