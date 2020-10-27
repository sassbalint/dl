# -*- coding: utf-8 -*-
"""
Test a model created by model_from_scratch.py
Provide a test sentence on stdin with a <mask> token
at position you want to guess.
"""


import argparse
import sys


# handle command line arguments
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--language', '-l',
    choices=['eo', 'hu'],
    help='select language',
    type=str,
    default='eo')
args = parser.parse_args()


LANGUAGE = args.language

# original Esperanto data truncated to 400_000 sentences
CORPUS = "EsperBERTo/oscar.eo.400000.txt"
OUTDIR = "EsperBERTo"

# Hungarian data -- 1_000_000 sentences
if LANGUAGE == 'hu':
    CORPUS = "/data/language/hungarian/corp/test/BL/mini/train_mini.txt"
    OUTDIR = "huBERTo"

print()
print('*** CONFIG ***')
print(f'language = {LANGUAGE}')
print(f'corpus = {CORPUS}')
print(f'outdir = {OUTDIR}')
print()

from transformers import pipeline

print('Loading model...')
fill_mask = pipeline(
    "fill-mask",
    model=OUTDIR,
    tokenizer=OUTDIR
)
print('Ready.')
print()

for sentence in sys.stdin:
    print(fill_mask(sentence))
    for guess in fill_mask(sentence):
        print(sentence.replace('<mask>', guess['token_str']))

