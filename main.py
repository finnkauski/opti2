# third party
import fastText as ft

# project
from optimus.optimus import run
from optimus.core.select import select

# print("[INFO] Loading fastText model")
# model = ft.load_model("wiki.en.bin")

results = run("tests/resources/example.txt", model, stepsize=2)
