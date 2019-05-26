# third party
import fastText as ft

# project
from optimus.core import parse
from optimus.core.embedding.fasttext import embed
from optimus.core.cluster import cluster

with open("tests/resources/example.txt") as handle:
    example = handle.read().splitlines()

# parse
target = list(parse.parse(parse.default_cleanerM(example)))

# cluster
numclusters = len(target)
startdepth = 3
enddepth = 15
step = 3
labels = {}

model = ft.load_model("wiki.en.bin")
embeddings = list(embed(target, model))
