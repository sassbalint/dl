# -*- coding: utf-8 -*-
"""
Test a model created by model_from_scratch.py
Provide test text on stdin with a <mask> token
at position you want to guess.
"""


import argparse
import sys


MASK_TOKEN = '<mask>'
SPACE_CHAR = 'Ġ' # RoBERTa space char


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
    default='eo'
)
parser.add_argument(
    '--iterative', '-i',
    help='iteratively predict next n words, instead of providing best guesses',
    type=int,
    default=1
)

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
from transformers.pipelines import PipelineException

print('Loading model...')
fill_mask = pipeline(
    "fill-mask",
    model=OUTDIR,
    tokenizer=OUTDIR
)
print('Ready.')
print()


def replace_and_print(text, predicted, numbering=False):
    replacer = predicted['token_str'].replace(SPACE_CHAR, ' ')
    text = text.replace(MASK_TOKEN, replacer)
    number = f'{i} ' if numbering else ''
    print(f'{number}[{text}]')
    return text


for text in sys.stdin:

    if args.iterative != 1: # iteration
        print(f' *** {args.iterative} iterative continuations:')
    else: # best guesses
        print(f' *** best guesses:')

    for i in range(args.iterative):
        try:
            predicteds = fill_mask(text)
        except PipelineException:
            print(f'Error in input. A {MASK_TOKEN} token is mandatory.')
            continue

        replacer = ''
        if args.iterative != 1: # iteration
            text = replace_and_print(text, predicteds[0], numbering=True)
        else: # best guesses
            print(predicteds)
            for predicted in predicteds:
                replace_and_print(text, predicted)

        # extend text with a MASK_TOKEN at the end for --iterative
        text += f'{MASK_TOKEN}'

    print()

