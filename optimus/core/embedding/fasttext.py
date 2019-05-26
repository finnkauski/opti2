"""
fasttext embedding functionality
================================

Embedding functionality to go from STR to [FLOAT] using fastText

"""
# stdlib
import toolz as fp

# third party
import toml


def embed(texts, model):
    """
    Embedding of a list of texts through fastText.

    In essence it uses the ``model.get_word_vector``
    and just maps across a list of texts

    Parameters
    ----------
    texts : list[str]
        the texts to be embedded

    model : fastText.FastText._FastText
        loaded fast text model

    Returns
    -------
    map
    """
    return map(model.get_word_vector, texts)
