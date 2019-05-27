# stdlib
import re

from math import log
from difflib import SequenceMatcher

# third party
import numpy as np
from textacy.similarity import jaccard, levenshtein

# project
from optimus.utils import strip, unpack


def characters(xs, start=3, finish=20):
    """
    Get charactergrams between set lengths

    Parameters
    ----------
    xs : list[str]
        list of texts to process

    start : int
        min length for ngram

    finish : int
        max length of ngram

    Returns
    -------
    list[str]
    """
    return unpack([get(x, n) for x in xs for n in range(start, finish)])


def get(x, n):
    """
    Fetch all charactergrams of length n from x

    Parameters
    ----------
    x : str
        string to get ngrams from

    n : int
        length of ngrams

    Returns
    -------
    list[str]
    """
    regex = f"(?=({'.'*n}))"
    return re.findall(regex, x)


def scorer(xs):
    """
    Calculate maximum score across texts

    Parameters
    ----------
    xs : list[str]
        list of texts to calculate scores for

    Returns
    -------
    float
        maximum score across the list of texts
    """
    cgrams = characters(xs)
    score = (sum(i in j for j in xs) ** 2 * (1 + log(len(i))) for i in cgrams)
    return max(score)


# TODO make sure this works when 2 have the same number on the score
def label(xs):
    """
    Pick the label that has the maximum score

    Parameters
    ----------
    xs : list[str]
        list of texts that will be processed and the
        highest score returned

    Returns
    -------
    float
    """
    cgrams = characters(xs)
    score = [np.sum(i in j for j in xs) ** 2 * (1 + log(len(i))) for i in cgrams]
    candidate = [cgrams[i] for i, j in enumerate(score) if j == np.max(score)]
    return candidate[0]


# +TODO: brain not working, make this betterer


# charactergram measurement
def condition(xs):
    """
    Find the longest common ngram metric

    Parameters
    ----------
    xs : list[str]
        list of texts to use for ngram finding
    Returns
    -------
    float
    """
    xs_ = strip(xs)
    return (
        0.0
        if ((len(xs_) <= 1) or (np.min([len(x) for x in xs_]) < 3))
        # TODO: if there is only one item then its vacuously 0?
        else scorer(xs_)
    )


def selection(xs):
    """
    Selecting the text which has most similarity with all other texts
    through chargram similarity

    Parameters
    ----------
    xs : list[str]
        text to be selected from

    Returns
    -------
    str
    """
    xs_ = strip(xs)
    return label(xs_)
