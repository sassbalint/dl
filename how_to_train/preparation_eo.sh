#!/bin/bash

# Install `transformers` from master
# pip install git+https://github.com/huggingface/transformers
# XXX nyilván, ha nincs; virtual env; requirements.txt ilyesmi...

DIR=EsperBERTo
mkdir -p $DIR

# In this notebook we'll only get one of the files (the Oscar one)
# for the sake of simplicity and performance
wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt
# truncate the file to 400_000 sentences
cat oscar.eo.txt | head -400000 > oscar.eo.400000.txt
mv oscar.eo* $DIR

# Check that we have a GPU
nvidia-smi
# Check that PyTorch sees it
python3 -c "import torch; print(torch.cuda.is_available())"

