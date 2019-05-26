from optimus.core import parse

with open("example_Strings.txt") as handle:
    example = handle.read().splitlines()

parsed = list(parse.parse(parse.default_cleanerM(example)))
