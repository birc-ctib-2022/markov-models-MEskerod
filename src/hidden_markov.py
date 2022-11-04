import numpy as np
from math import log

class MarkovModelHidden:
    """Representation of a Markov model."""

    init_probs: list[float]
    trans: list[list[float]]
    emit_probs: list[list[float]]

    def __init__(self,
                 init_probs: list[float],
                 trans: list[list[float]], emit_probs: list[list[float]]):
        """Create model from initial and transition probabilities."""
        # Sanity check...
        k = len(init_probs)
        assert k == len(trans)
        for row in trans:
            assert k == len(row)
        assert k == len(emit_probs)

        self.init_probs = init_probs
        self.trans = trans
        self.emit_probs = emit_probs


def joint_prob(Z: list[int], X: list[int], mm: MarkovModelHidden) -> float:
    """
    Compute the likelihood of mm given x.

    This is the same as the probability of x given mm,
    i.e., P(x ; mm).

    X are the observed and Z is the hidden sequence

    What is the probability of observing X given Z 
    """
    assert len(Z) == len(X)
    if not Z:
        return 1

    probablity = mm.init_probs[Z[0]] * mm.emit_probs[Z[0]][X[0]]
    for i in range(1, len(Z)): 
        probablity *= mm.trans[Z[i-1]][Z[i]] * mm.emit_probs[Z[i]][X[i]]
    return probablity

def log_joint_prob(Z: list[int], X: list[int], mm: MarkovModelHidden) -> float: 
    assert len(Z) == len(X)
    if not Z: 
        return log(1)
    probability = log(mm.init_probs[Z[0]]) + log(mm.emit_probs[Z[0]][X[0]])
    for i in range(1, len(Z)):
        probability += log(mm.trans[Z[i-1]][Z[i]]) + log(mm.emit_probs[Z[i]][X[i]])
    return probability

