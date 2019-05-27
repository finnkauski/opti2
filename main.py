# third party
import fastText as ft

# project
from optimus.optimus import run

model = ft.load_model("wiki.en.bin")

results = run("tests/resources/example.txt", model)
