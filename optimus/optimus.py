# stdlib
# third party
import pandas as pd
import fastText as ft

# project
from optimus.core import parse
from optimus.core.embedding.fasttext import embed
from optimus.core.cluster import cluster
from optimus.core.select import select

# TODO: REWRITE TIER NAMES, make the step size more granular


def run(data, model, depth=3, end_depth=15, stepsize=3, **thresholds):
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

    depth : int
        the initial depth to start at for the cluster cut off

    end_depth : int
        the final depth to stop at

    stepsize : int
        the stepsize to increment by

    **thresholds
        the thresholds that a user can provide to the pipeline
        for label selection. See the docs for optimus.core.select
        valid: edits, wordgram, chargram, hypernym

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
    print("[START] Autobots, roll out!")
    print("    -- parsing")
    target = list(parse.parse(parse.default_cleanerM(data)))

    targetM = target.copy()  # mutable target

    # get number of initial clusters
    numclusters = len(target)

    # embed
    print("    -- embedding")
    embeddings = list(embed(targetM, model))

    # cluster
    print("    -- clustering")
    clusters = cluster(embeddings, depth)

    # main loop
    while depth <= end_depth:
        print(f"** Depth {depth}")
        print("    -- embedding")
        embedding = list(embed(targetM, model))

        print("    -- clustering")
        clusters = cluster(embedding, depth)

        if len(set(clusters)) == numclusters:
            print("    >> No new clusters generated")
            labels[f"tier_{depth}"] = labels[f"tier_{depth-stepsize}"]
        else:
            numclusters = len(set(clusters))

            print("    -- generating labels")
            labels[f"tier_{depth}"] = list(select(targetM, clusters))
            targetM = [i if i else j for i, j in zip(labels[f"tier_{depth}"], targetM)]

        depth += stepsize

    # make the output match optimus
    dty = labels
    dty["original"] = data
    dty["current_labels"] = dty[f"tier_{end_depth}"]

    df = pd.DataFrame.from_dict(dty, orient="index").transpose()

    # reordering to match optimus
    df_ = df[[df.columns[-2], *list(df.columns[:-3]), df.columns[-1]]]

    return df
