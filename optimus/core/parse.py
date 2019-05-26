# stdlib
import re

# third party
import toml
import spacy as sp
import toolz as fp
import pandas as pd

# load spacy model
print("[WARN] Loading `en_core_web_md` model from spacy. Might take a few seconds.")
nlp = sp.load("en_core_web_md")

# load in default_regex strings
with open("optimus/etc/regexes") as handle:
    replace = toml.load(handle)

# helper for regex
def default_cleaner(string, regex_dict=replace):
    """
    default_cleaner(string, regex_dict={"regex":"replacement"})

    A default cleaner for text. The goal for this is to remove
    the unnecessary words and other things such as numbers from the
    text. The default dictionary for this is in the ``optimus/etc/regexes``
    file.

    A version of this function that maps across a list exists
    and is named ``default_cleanerM``

    Parameters
    ----------
    string : str
        the string to be cleaned

    regex_dict : dict
        dictionary containing the replacement
        regex and the thing to replace it with

    Returns
    -------
    str
    """
    return fp.compose(
        *[fp.partial(re.sub, i, j, flags=re.IGNORECASE) for i, j in regex_dict.items()]
    )(string)


default_cleanerM = fp.partial(map, default_cleaner)

# spacy based parsing
def keyword(document):
    """
    Find the dependant subject of the document.

    Parameters
    ----------

    document : spacy.tokens.doc.Doc
        spacy document to extract the subject from

    Returns
    -------
    str
    """
    return [token for token in filter(lambda x: x.dep_ == "ROOT", document)][0]


keywordM = fp.partial(map, keyword)


def lemma(token) -> str:
    """
    Find the lemma of a given token.

    Parameters
    ----------

    token : spacy.tokens.token.Token
        the individual token from a spacy document

    Returns
    -------
    str
        the lemma of the provided token
    """
    return token.lemma_


def vocab(word):
    """
    Check if the word is in the vocab of the model

    Parameters
    ----------
    word : str

    Returns
    -------
    str
        it returns the original word or an empty string
        if the word is not in the vocabulary
    """
    return word if word in nlp.vocab else ""


parse_ = fp.compose(vocab, lemma, keyword, nlp)


def parse(texts):
    """
    Map the parsing function across the list of texts.

    The parsing function currently is:
    - pass through a spacy model (nlp)
    - get the keyword of the sentence (keyword)
    - lemma the resulting word (lemma)
    - vocab check that the word is in the vocabulary (vocab)

    For the single document version use: ``parse_``

    Parameters
    ----------
    texts : list
        list of text strings

    Returns
    -------
    map object
    """
    return map(parse_, texts)
