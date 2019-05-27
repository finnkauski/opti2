# third party
import numpy as np

from textacy.similarity import jaccard

# project
from optimus.utils import strip


def fetch(x, n):
    """
    Fetch ngrams (supporting function for wordgrams)

    Parameters
    ----------
    x : str
        text string

    n : int
        ngram length

    Returns
    -------
    set
    """
    tokens = x.split()
    ngrams = (" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return set(ngrams)


def group(x):
    """
    Supporting function for worgrams

    Parameters
    ----------
    x : str
        text to group

    Returns
    -------
    set
    """
    return (
        x and set.union(*(fetch(x, i) for i in range(1, len(x.split()) + 1)))
    ) or set()


def wordLen(xs):
    """
    Get minimum word number in a text

    Parameters
    ----------
    xs : list[str]
        list of texts to calculate minumum distance on

    Returns
    -------
    int
        minimum word number for all texts in list
    """
    tokens = map(str.split, xs)
    return np.min([len(w) for t in tokens for w in t])


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
    score = (sum(i in j for j in xs) ** 2 * (1 + np.log(len(i.split()))) for i in xs)
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
    ys = [j for i in xs for j in group(i)]
    score = (sum(i == j for j in ys) ** 2 * (1 + np.log(len(i.split()))) for i in ys)
    nScore = (i / len(ys) for i in score)
    return ys[nScore == max(nScore)]


# TODO: this requires going over the list a couple of times, its pretty fast but
#      could be better - if it becomes a resource hog then fix it
# TODO: need to change score so that it stops counting against itself and the
#      string it came from as this weighs words from long descriptions more

# wordgram measurement
def condition(xs):
    """
    Average number of words across a cluster

    Parameters
    ----------
    xs : list[str]
         list of texts to calculate the wordgrams on

    Returns
    -------
    float
    """
    xs_ = strip(xs)
    return 0.0 if (len(xs_) <= 1) or (wordLen(xs_) < 3) else scorer(xs_)


def selection(xs):
    """
    Selecting the text which has most similarity with all other texts
    through wordgram similarity

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
