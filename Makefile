.DEFAULT_GOAL := tests
.PHONY: tests
tests:
	@python -m pytest tests
