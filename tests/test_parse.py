# project
from optimus.core import parse

# load in example strings
with open("tests/resources/example.txt") as handle:
    example = handle.read().splitlines()


def test_parse():
    parsed = parse.parse(example)

    # check that the length was not altered
    assert len(example) == len(parsed)
    # check that all are strings
    assert all(map(lambda entry: isinstance(entry, str), parsed))
