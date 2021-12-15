from ..hparams import args

import numpy as np
from scipy.special import softmax
import torch
from torch.distributions.categorical import Categorical


def init_Q(A):
    Q = (0 if args["q"].startswith("max") else 2) * np.ones((A, A, 2))
    return Q


def update(Q, arc, rew):
    extreme = max if args["q"].startswith("max") else min

    prev = None
    for _, layer in arc.items():
        if prev is not None:
            i = int(prev[0].item())
            j = int(layer[0].item())
            Q[i, j, 1] = extreme(Q[i, j, 1], rew)

        op_id, skips = layer[0], layer[1]
        for idx, c in enumerate(skips):
            i = arc[str(idx)][0].item()
            j = int(op_id.item())

            Q[i, j, int(c)] = extreme(Q[i, j, int(c)], rew)
        prev = layer

    return Q


def sample(Q, arc_seq, sandwich=False):
    D = len(Q)

    cumulative = np.zeros(D)
    for l in arc_seq:
        row = np.mean(Q[l[0]], axis=1)
        if sandwich:
            row = softmax(row)
        cumulative += row

    prob_dist = softmax(cumulative, axis=0)
    prob_dist = torch.from_numpy(prob_dist)

    branch_id_dist = Categorical(logits=prob_dist)
    branch_id = torch.tensor([branch_id_dist.sample()])

    arc_seq.append([branch_id, []])

    for l in arc_seq[:-1]:
        row = Q[l[0]][branch_id.item()]
        row = torch.from_numpy(row)

        skip_dist = Categorical(logits=row)
        arc_seq[-1][1].append(skip_dist.sample())

    arc_seq[-1][1] = torch.tensor(arc_seq[-1][1])

    return arc_seq
