# stdlib
# third party
import fastText as ft

# project
from optimus.core import parse
from optimus.core.embedding.fasttext import embed
from optimus.core.cluster import cluster

# setup config


def run(data, model, sdepth=3, edepth=15, stepsize=3):
    """
    The main pipeline for optimus. All the parts for this can be found
    individually, but this function executes the optimus pipeline.

    Parameters
    ----------
    data : list[str] | str
        can either be a list of strings to be processed or a string path
        to the text file containing the data

    model : fastText.FastText._FastText
        fastText model to be used for embedding
        in future other models will be supported

    Returns
    -------
    pandas.core.frame.DataFrame
        pandas dataframe with original results and the different iterations
        optimus
    """

    # prepare final dict
    labels = {}

    # overwrite data if filepath string is provided
    if isinstance(data, str):
        with open(data) as handle:
            data = handle.read().splitlines()

    # parse the data
    target = list(parse.parse(parse.default_cleanerM(data)))

    # get number of initial clusters
    numclusters = len(target)

    # embed
    embeddings = list(embed(target, model))

    # cluster
    clusters = cluster(embeddings, sdepth)

    return clusters
