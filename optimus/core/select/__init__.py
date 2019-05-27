# third party
import toml
import spacy
import numpy as np
import toolz as fp

# project
from optimus.core.select import edits as ed
from optimus.core.select import wordgrams as wg
from optimus.core.select import chargrams as cg
from optimus.core.select import hypernyms as hn


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


def build_represent(edit=0.75, wordgram=2.00, chargram=0.8, hypernym=0.00):
    """
    Given a set of thresholds (edit, wordgram, chargram, hypernym)
    return a function that would get the right represenatation for the
    cluster.

    Parameters
    ----------
    edit : float
        float for thresholding edit distance

    wordgram : float
        float for thresholding wordgrams

    chargram : float
        float for thresholding charactergrams

    hypernym : float
        float for thresholding hypernym wordnet

    Returns
    -------
    function
    """
    # exposed default interfaces
    def represent_(decisions, functions, group):
        for d, f in zip(decisions, functions):
            if d(group):
                return f(group)  # stops the first time that d(group) is true

    decisions = (
        lambda xs: ed.condition(xs) > edit,  # high lexical similarity
        lambda xs: wg.condition(xs) > wordgram,  # medium lexical similarity
        lambda xs: cg.condition(xs) > chargram,  # low lexical similarity
        lambda xs: hn.condition(xs) > hypernym,  # semantic similarity
        lambda xs: True,  # default always true fallback
    )

    functions = (
        lambda xs: ed.selection(xs),  # high lexical similarity
        lambda xs: wg.selection(xs),  # medium lexical similarity
        lambda xs: cg.selection(xs),  # low lexical similarity
        lambda xs: hn.selection(xs),  # semantic similarity
        lambda xs: fallback(xs),  # default fallback
    )

    return fp.partial(represent_, decisions, functions)


def select(xs, clusters, **thresholds):
    """
    Select appropriate label using the labelling steps
    of edit distance -> wordgrams -> chargrams -> wordnet

    Parameters
    ----------
    xs : list[str]
        list of texts to work through

    cluster : list[int]
        list of cluster numbers corresponding to where the
        existing label is allocated

    **thresholds
        you can pass custom threshold values here to alter the
        default cutoffs.

        valid values are:

        edit - thresholding edit distance (default: 0.75)

        wordgram - float for thresholding wordgrams (default: 2.0)

        chargram - float for thresholding charactergrams (default: 0.8)

        hypernym - float for thresholding hypernym wordnet (default: 0.0)

    Returns
    -------
    list[str]
        labels assigned through the selection procedure
    """
    represent = build_represent(**thresholds)
    groups = build(xs, clusters)
    labels = [represent(group) for group in groups]
    return (labels[cluster - 1] for cluster in clusters)
