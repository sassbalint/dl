
SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=scripts
D=data


# ===== MAIN STUFF

PATH_PREFIX=$D/mnsz2_pos/mnsz2_330000_POS_
SUBSAMPLE_SIZE=1000
SCRIPT=$S/bert_sklearn_tokseq_finetune.py
finetune:
	date ; time python3 $(SCRIPT) -p $(PATH_PREFIX) -s $(SUBSAMPLE_SIZE); date


# ===== utils

prepare: install getdata

# MNSZ2 POS data
DATASOURCE=http://corpus.nytud.hu/people/joker/mnsz2_330000_POS_
getdata:
	mkdir -p $D
	cd $D ; rm -rf mnsz2_pos ; mkdir -p mnsz2_pos ; cd mnsz2_pos ; wget $(DATASOURCE)train.txt ; wget $(DATASOURCE)dev.txt ; wget $(DATASOURCE)test.txt

# torch max. verzió: 1.4
install:
	#conda activate torch1.4 -- KÉZZEL KELL!
	nvidia-smi
	rm -rf bert-sklearn
	git clone -b master https://github.com/charles9n/bert-sklearn
	cd bert-sklearn; pip install .

