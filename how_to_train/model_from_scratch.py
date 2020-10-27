# -*- coding: utf-8 -*-
"""
How to train a new language model
from scratch using Transformers and Tokenizers?
https://huggingface.co/blog/how-to-train
Original notebook from:
https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb
This version is adapted for a NYTI borhűtő.
"""


import argparse


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
EXAMPLE_SENTENCES = [
    "La suno <mask>.",
    "Jen la komenco de bela <mask>."
]

# Hungarian data -- 1_000_000 sentences
if LANGUAGE == 'hu':
    CORPUS = "/data/language/hungarian/corp/test/BL/mini/train_mini.txt"
    OUTDIR = "huBERTo"
    EXAMPLE_SENTENCES = [
        "Ez egy <mask>.",
        "Most kell <mask>.",
        "Nem fog <mask>.",
        "Ez egy <mask> könyv."
    ]

print()
print('*** CONFIG ***')
print(f'language = {LANGUAGE}')
print(f'corpus = {CORPUS}')
print(f'outdir = {OUTDIR}')
print(f'example_sentences = {EXAMPLE_SENTENCES}')
print()


def msg(msg):
    print()
    print(msg)


msg(f'1. (Just a placeholder for running preparation_{LANGUAGE}.sh)')
## 1. Run preparation.sh
# to install some stuff, download dataset, whatever


msg("2. Train a tokenizer")
## 2. Train a tokenizer

from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(
    files=CORPUS,
    vocab_size=52_000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
])

tokenizer.save_model(OUTDIR)

print("vocab.json")
with open(f'{OUTDIR}/vocab.json') as f:
    for line in f: # only the first line is needed (there is only 1)
        elems = line.split(',')[0:10] # "process json"
        for elem in elems:
            print(elem)

from itertools import islice

print("merges.txt")
with open(f'{OUTDIR}/merges.txt') as f:
    for line in islice(f, 10):
        print(line, end='')


msg("3. Train a language model from scratch")
## 3. Train a language model from scratch
# We use `Trainer` directly.
# As the model is BERT-like, we’ll train it on a task
# of Masked language modeling, i.e. the predict how to fill
# arbitrary tokens that we randomly mask in the dataset.


msg("3.1 We'll define the following config for the model")
### 3.1 We'll define the following config for the model

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


msg("3.2 Now let's re-create our tokenizer in transformers")
### 3.2 Now let's re-create our tokenizer in transformers

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(
    OUTDIR, max_len=512)


msg("3.3 Finally let's initialize our model")
### 3.3 Finally let's initialize our model
# As we are training from scratch,
# we only initialize from a config,
# NOT from an existing pretrained model or checkpoint.

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

print(f'number of parameters = {model.num_parameters()}')
# => 84 million parameters


msg("3.4 Now let's build our training Dataset")
### 3.4 Now let's build our training Dataset
# by applying our tokenizer to our text file.

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=CORPUS,
    block_size=128,
)


msg("3.5 data_collator")
### 3.5 data_collator -- ez mi lehet? XXX

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


msg("3.6 Finally, we are all set to initialize our Trainer")
### 3.6 Finally, we are all set to initialize our Trainer

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTDIR,
    overwrite_output_dir=True,
    num_train_epochs=1,

    # https://github.com/pytorch/pytorch/issues/16417#issuecomment-497952224
    # -> valszeg a 64 sok lesz, nézzük 32-vel! MEGY! :)
    per_gpu_train_batch_size=32,

    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)


msg("3.7 Start training")
### 3.7 Start training

trainer.train()

# XXX 3 hours (!) on colab
# XXX in the meantime look at: nvidia-smi


msg("3.8 🎉 Save final model (+ tokenizer + config) to disk")
### 3.8 🎉 Save final model (+ tokenizer + config) to disk

trainer.save_model(OUTDIR)


msg("4. Check that the LM actually trained")
## 4. Check that the LM actually trained
# to check whether our language model is learning
# anything interesting is via the `FillMaskPipeline`.
# There are other interesgint pipelines as well!
# see: https://huggingface.co/transformers/main_classes/pipelines.html

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model=OUTDIR,
    tokenizer=OUTDIR
)

for sentence in EXAMPLE_SENTENCES:
    print(fill_mask(sentence))

# XXX XXX XXX és itt jöhet a blog (!) szerinti finetuning!
# XXX XXX XXX és itt jöhet a blog (!) szerinti finetuning!
# XXX XXX XXX és itt jöhet a blog (!) szerinti finetuning!

