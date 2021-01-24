
SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=scripts
D=data


# ===== MAIN STUFF


# MNSZ2 frag data
DATAFILE=mnsz2_660000_frag
#DATAFILE=mnsz2_330000_frag
# MNSZ2 POS data
#DATAFILE=mnsz2_330000_POS

PATH_PREFIX=$D/mnsz2_pos/$(DATAFILE)_
SUBSAMPLE_SIZE=1000
SCRIPT=$S/bert_sklearn_tokseq_finetune.py
finetune:
	date ; time python3 $(SCRIPT) -p $(PATH_PREFIX) -s $(SUBSAMPLE_SIZE); date


# ===== utils

prepare: install getdata

DATASOURCE=http://corpus.nytud.hu/people/joker/$(DATAFILE)_

getdata:
	mkdir -p $D
	rm -f $(PATH_PREFIX)* ; cd $D ; mkdir -p mnsz2_pos ; cd mnsz2_pos ; wget $(DATASOURCE)train.txt ; wget $(DATASOURCE)dev.txt ; wget $(DATASOURCE)test.txt

# torch max. verzió: 1.4
install:
	@echo "ensure that pytorch version is <=1.4.0 by e.g. 'conda activate torch1.4'"
	nvidia-smi
	rm -rf bert-sklearn
	git clone -b master https://github.com/charles9n/bert-sklearn
	cd bert-sklearn; pip install .

