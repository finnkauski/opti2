# base
import itertools

# third party
import spacy
import numpy as np

from textacy.similarity import jaccard
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# project
from optimus.utils import strip
from optimus.core.parse import keyword, nlp

model = spacy.load("en_core_web_md")
model.add_pipe(WordnetAnnotator(model.lang), after="tagger")

# semantic similarity
def condition(xs):
    """
    Implement hypernym lookup

    Parameters
    ----------
    xs : list[str]
        list of texts

    Returns
    -------
    float
    """
    xs_ = strip(xs)
    return 0.0 if len(xs_) <= 1 else 1.0


def selection(xs):
    """
    Select using hypernyms

    Parameters
    ----------
    xs : list[str]
        list of texts to select from

    Returns
    -------
    str
    """
    return hypernyms(xs)


def hypernyms(
    xs, avoided_hypernyms=("entity", "whole", "object", "matter", "physical_entity")
):
    """
    Given a list of words/texts find a common hypernym

    Parameters
    ----------
    xs : list[str]
        list of texts

    Returns
    -------
    str
        common hypernym
    """
    corpus = map(keyword, gencorpus(xs))

    syns = [token._.wordnet.synsets() for token in corpus]
    lengths = [len(s) for s in syns]
    flatM = [j for i in syns for j in i]

    common = []
    for length in lengths:
        left = flatM[:length]
        right = flatM[length:]

        for i, j in itertools.product(left, right):
            common.append(lca(i, j))
        flatM = right

    # remove things which are too abstract
    common = [j.name().split(".")[0] for i in common for j in i]
    common = [i for i in common if i not in avoided_hypernyms]

    # if it cannot find a common hypernym returns empty string
    return (len(common) and common[0].replace("_", " ")) or ""


def gendoc(x):
    """
    Make spacy document from a string

    Parameters
    ----------
    x : str

    Returns
    -------
    spacy.tokens.doc.Doc
    """
    return model(str(x))


def gencorpus(xs):
    """
    Map the gendoc function across a list to generate full corpus

    Parameters
    ----------
    xs : list[str]

    Returns
    -------
    list [spacy.tokens.doc.Doc]
    """
    return map(gendoc, xs)


def lca(syn1, syn2):
    """
    Find lowest common ancestor
   
    """
    return syn1.lowest_common_hypernyms(syn2)
