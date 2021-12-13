import numpy as np
from scipy.special import softmax
from ..hparams import args


def init_Q(D):
    return np.zeros((D, D))


def update(Q, ops, rew, D=2, max=True):
    ops = [int(op) for op in ops]
    D = min(D, len(Q.shape))

    ops = list(set(ops))
    D1 = ops
    D2 = np.array([ops] * len(ops)).T
    extreme = np.maximum if args["q"] == "max" else np.minimum
    Q[D1, D2] = extreme(Q[D1, D2], rew)

    return Q


def sample(Q, arc_seq, sandwich=False):
    D = len(Q)
    cumulative = np.zeros(D)
    for a in arc_seq:
        row = Q[a[0]]
        if sandwich:
            row = softmax(row)
        cumulative += row

    prob_dist = softmax(cumulative)
    return prob_dist
