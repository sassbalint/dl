# how to train a new language model from scratch

# basics

source: https://huggingface.co/blog/how-to-train + https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb

* slightly modified / improved
* adapted for Hungarian
* convenient `Makefile`
* separate `test_model.py` script

# installation

```bash
pip install tokenizers
pip install git+https://github.com/huggingface/transformers
nvcc --version
```

Last line tells you version of CUDA.
Use it in the next line which is
officially recommended by https://pytorch.org.

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Test the installation by:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

## troubleshooting

__If__ you encounter segfault on `import transformers`,
your `torch` [may be too old.](https://github.com/huggingface/transformers/issues/118)
In this case type:

```bash
pip show torch | grep Version
```

This tells you version of `pytorch`.
I think `1.5` is needed at least.
So if it is less then:

```bash
pip uninstall torch
pip install torch==1.7
```

__If__ you encounter `RuntimeError: CUDA out of memory....`,
`batch_size` [may be too large.](https://github.com/pytorch/pytorch/issues/16417#issuecomment-4)\
Edit `model_from_scratch.py` and
change `per_gpu_train_batch_size` in _half_ and try again.

# usage

## training

To train a Hungarian `RoBERTa` model.

```bash
make LANGUAGE=hu train_model_from_scratch
```

Running time: 75 min.

## testing

To test the model by masked examples from stdin,
e.g. _"Ez egy \<mask\> könyv."_

```bash
make LANGUAGE=hu test_model
```

