"""
Finetune for text classification.
"""

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np

import sys

if not(len(sys.argv) >= 2 and sys.argv[1] in {'glue-cola', 'imdb'}):
    print("Command line param should be `glue-cola` or `imdb`")
    exit(1)

if sys.argv[1] == 'glue-cola':

    # https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb
    from transformers import AutoModelForSequenceClassification

    MODEL = "distilbert-base-uncased"
    TASK = ("glue", "cola") # GLUE's CoLA task = grammatical or not?
    # keys: 'idx', 'label', 'sentence'
    DATACOLUMN = "sentence"
    # new keys after tok: 'attention_mask', 'input_ids'
    TRAIN = "train"
    EVAL = "validation"
    METRIC = TASK
    SAVE_DIR = f'{MODEL}-finetuned-{"-".join(TASK)}'

    DATASET_SIZE = 0 # 0 means all data
    BATCH_SIZE = 32 # based on gpustat -cupF

elif sys.argv[1] == 'imdb':

    # https://huggingface.co/docs/transformers/training / COLAB notebook
    from transformers import AutoModelForSequenceClassification

    MODEL = "bert-base-cased"
    TASK = ("imdb",) # sentiment on movie reviews
    # keys: 'text', 'label'
    DATACOLUMN = "text"
    # new keys after tok: 'attention_mask', 'input_ids', 'token_type_ids'
    TRAIN = "train"
    EVAL = "test"
    METRIC = ("accuracy",)
    SAVE_DIR = "test-trainer"

    DATASET_SIZE = 500 # 0 means all data
    BATCH_SIZE = 4 # based on gpustat -cupF


def section(s): print(f'\n>> {s} <<\n')
def msg(msg, obj, inline=False):
    sep = ': ' if inline else '\n'
    print(f'** {msg}{sep}{obj}\n')

section("load dataset")
raw_datasets = load_dataset(*TASK)
msg('raw datasets', raw_datasets)
msg('raw/train/0', raw_datasets[TRAIN][0])

section("tokenize dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL) # use_fast=True

example = raw_datasets[TRAIN][4]
tokenized_example = tokenizer(example[DATACOLUMN])
tokens = tokenizer.convert_ids_to_tokens(tokenized_example["input_ids"])
msg('ex', example)
msg('ex/subwords', tokens)

def tokenize_function(examples):
    return tokenizer(examples[DATACOLUMN], truncation=True)

msg('tokenized', tokenize_function(raw_datasets['train'][:5]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# az első 20 subword-ID a neki megfelelő ("dekódolt") stringgel
for t in tokenized_datasets['train'][0]['input_ids'][:20]:
    print(t, tokenizer.decode(t))

msg('tokenized datasets', tokenized_datasets)

if DATASET_SIZE > 0:
    train_dataset = tokenized_datasets[TRAIN].shuffle(seed=42).select(range(DATASET_SIZE))
    eval_dataset = tokenized_datasets[EVAL].shuffle(seed=42).select(range(DATASET_SIZE))
else:
    train_dataset = tokenized_datasets[TRAIN]
    eval_dataset = tokenized_datasets[EVAL]

section("load model")
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

args = TrainingArguments(
    SAVE_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=5,

    #save_strategy = "epoch",
    #load_best_model_at_end=True,
    #metric_for_best_model="...",
)

section("metric")
metric = load_metric(*METRIC)

import random
fake_labels = np.random.randint(0, 2, size=(64,))
fake_preds = np.random.randint(0, 2, size=(64,))
msg('fake labels', fake_labels)
msg('fake preds', fake_preds)
msg(f'example `{METRIC}` metric calculation', metric.compute(predictions=fake_preds, references=fake_labels))

def logits2preds(logits):
    return np.argmax(logits, axis=-1) # last dimension

# XXX how to compute many metrics (P, R, F1...)?
# 2-dim labels+preds: sentence, value <- SequenceClassification
def compute_metrics(eval_pred):
    predicted_logits, labels = eval_pred
    predictions = logits2preds(predicted_logits)
    results = metric.compute(predictions=predictions, references=labels)
    return results

section("finetune")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer, # needed? for a default data collator?
    compute_metrics=compute_metrics,
)
trainer.train()

# how to obtain predictions?
# -> https://huggingface.co/course/chapter3/3?fw=pt / predict
section("predict")
preds, labels, _ = trainer.predict(eval_dataset)
ok = 0
for i, (et, pp, pl) in enumerate(zip(eval_dataset[DATACOLUMN], preds, labels)):
    pred = logits2preds(pp)
    if pl == pred:
        ok += 1
        annot = ''
    else:
        annot = '*'
    # a szöveg eleje, valós címke, jósolt címke, '*' ha nem stimmel
    print(f'{i:5} {et[:40]:<40} {pl} {pred} {annot}')
# hány jó az összesből = egyeznie kell az evaluate() eredményével
msg('.', ok/len(labels))

section("eval")
result = trainer.evaluate()
msg('RESULT', result)

