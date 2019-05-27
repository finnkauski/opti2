# stdlib
import itertools

from difflib import SequenceMatcher

# third party
import toml
import spacy
import numpy as np
import toolz as fp

# project
from optimus.core.select import edits as ed


def build(xs, clusters):
    """
    Extract the strings by cluster number

    Parameters
    ----------
    xs : list[str]
        list of strings to be extracted from

    clusters : list[int]
        list of cluster labels

    Returns
    -------
    list[numpy.ndarray]
        list of numpy arrays each containing text from each
        cluster
    """
    return (np.extract(clusters == i, xs) for i in np.unique(clusters))


def fallback(xs):
    """
    A placeholder function as a fallback. If something can't find
    a suitable label this will be run and returned

    Parameters
    ----------
    xs : list[str]
        not required as anything passed in here is not used

    Returns
    -------
    str
        an empty string
    """
    return ""


# given conditions on the cluster and a function to run in each condition then
# do the following in a more abstract way
decisions = (
    lambda xs: ed.condition(xs) > config["edit"],  # high lexical similarity
    lambda xs: wg.condition(xs) > config["word"],  # medium lexical similarity
    lambda xs: cg.condition(xs) > config["char"],  # low lexical similarity
    lambda xs: hn.condition(xs) > config["hype"],  # semantic similarity
    lambda xs: True,  # default always true fallback
)

functions = (
    lambda xs: ed.selection(xs),  # high lexical similarity
    lambda xs: wg.selection(xs),  # medium lexical similarity
    lambda xs: cg.selection(xs),  # low lexical similarity
    lambda xs: hn.selection(xs),  # semantic similarity
    lambda xs: fallback(xs),  # default fallback
)

# exposed default interfaces
def represent_(decisions, functions, group):
    for d, f in zip(decisions, functions):
        if d(group):
            return f(group)  # stops the first time that d(group) is true


represent = functools.partial(represent_, decisions, functions)


def select_(xs: List[str], clusters: List[str], cluster: int) -> List[str]:
    groups = build(xs, clusters)
    labels = [represent(group) for group in groups]
    return labels[cluster - 1]


def select(xs: List[str], clusters: List[str]) -> List[str]:
    groups = build(xs, clusters)
    labels = [represent(group) for group in groups]
    return (labels[cluster - 1] for cluster in clusters)
