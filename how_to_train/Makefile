
SHELL := /bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"


# ===== MAIN STUFF

LANGUAGE=eo
ITERATIVE=5

prep:
	./preparation_$(LANGUAGE).sh

train_model_from_scratch:
	python3 model_from_scratch.py --language $(LANGUAGE)

# best guesses for `<mask>`
test_model:
	python3 test_model.py --language $(LANGUAGE)

# guess iteratively
test_model_iterative:
	python3 test_model.py --language $(LANGUAGE) --iterative $(ITERATIVE)

