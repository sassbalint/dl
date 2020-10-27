#!/bin/bash

# Install `transformers` from master
# pip install git+https://github.com/huggingface/transformers
# XXX nyilván, ha nincs; virtual env; requirements.txt ilyesmi...

DIR=huBERTo
mkdir -p $DIR

# Data is available by default. No need to download or whatever.

# Check that we have a GPU
nvidia-smi
# Check that PyTorch sees it
python3 -c "import torch; print(torch.cuda.is_available())"

