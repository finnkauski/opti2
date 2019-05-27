# stdlib
import itertools

# third party
import numpy as np

from textacy.similarity import levenshtein

# project
from optimus.utils import strip


# maps levenshtein distance into the closed interval [0,1]
def condition(xs):
    """
    Average edit distance on a cluster

    Parameters
    ----------
    xs : list[str]
         list of texts to calculate the edit distances from

    Returns
    -------
    float
    """
    xs_ = strip(xs)
    return (
        0.0  # levenshtein returns 1 if they are identical
        if len(xs_) <= 1
        else np.mean([levenshtein(*ys) for ys in itertools.combinations(xs_, 2)])
    )


def editMatrix(xs):
    """
    Create edit distance matrix

    Parameters
    ----------
    xs : list[str]
        text to create edit distance from

    Returns
    -------
    numpy.ndarray
        the distance array
    """
    return np.array([[levenshtein(x, y) for y in xs] for x in strip(xs)])


# +TODO: selection doesnt run if you have an empty list or just 1 empty string
def selection(xs):
    """
    Selecting the word which has most similarity with all other words

    Parameters
    ----------
    xs : list[str]
        text to be selected from

    Returns
    -------
    str
    """
    xs_ = strip(xs)
    return xs_[np.sum(editMatrix(xs_), axis=1).argmax()]
