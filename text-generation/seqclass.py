"""
https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb
"""

# pip install transformers datasets

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

import random

MODEL = "distilbert-base-uncased"
TASK = "cola" # GLUE' CoLA task = grammatical or not?
              # see other GLUE tasks in original notebook
BATCH_SIZE = 64 # based on gpustat -cp --watch

def msg(s): print(f'\n>> {s} <<\n')

msg("load dataset")
raw_datasets = load_dataset("glue", TASK) # 'idx', 'label', 'sentence'
DATACOLUMN = "sentence"
print(raw_datasets)
print(raw_datasets["train"][0])

msg("tokenize dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL) # use_fast=True

print(tokenizer("Hello, this one sentence!", "And this sentence goes with it."))
print(f'Sentence: {raw_datasets["train"][0][DATACOLUMN]}')

def tokenize_function(examples):
    return tokenizer(examples[DATACOLUMN], truncation=True)

print(tokenize_function(raw_datasets['train'][:5]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 'attention_mask', 'idx', 'input_ids', 'label', 'sentence' van ebben

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
# ha kisebb kell...
# train_dataset = tokenized_datasets["train"].shard(index=1, num_shards=10) 

msg("load model")
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

args = TrainingArguments(
    f'{MODEL.split("/")[-1]}-finetuned-{TASK}',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=5,
    weight_decay=0.01,

    # needed for `load_best_model_at_end`
    #save_strategy = "epoch",
    #load_best_model_at_end=True,
    #metric_for_best_model="matthews_correlation",

    # a hf access token is needed to do this
    #push_to_hub=True,
)

msg("metric")
metric = load_metric('glue', TASK)
#print(metric)
# metric.compute(your_predictions, labels)
# will return a dictionary with the metric(s) value:
fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
print(metric.compute(predictions=fake_preds, references=fake_labels))

def logits2preds(logits):
    return np.argmax(logits, axis=-1) # XXX WAS `1` SHOULD BE `-1`

# XXX how to compute many metrics (P, R, F1...)?
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits2preds(logits)
    return metric.compute(predictions=predictions, references=labels)

msg("finetune")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# how to obtain predictions?
# -> https://huggingface.co/course/chapter3/3?fw=pt / predict
msg("predict")
po = trainer.predict(eval_dataset)
# ha jól látom: eval_dataset['label'] == po.label_ids
# -> azaz predict() kimenetében az eredeti címkék vannak a .label_ids -ben
#    a predikciók a .predictions -ban vannak -- logit-ként!
ok = 0
for i, (et, pl, pp) in enumerate(zip(eval_dataset[DATACOLUMN], po.label_ids, po.predictions)):
    pred = logits2preds(pp)
    if pl == pred:
        ok += 1
        annot = ''
    else:
        annot = '*'
    # a szöveg eleje, valós címke, jósolt címke, '*' ha nem stimmel
    print(f'{i:5} {et[:40]:<40} {pl} {pred} {annot}')
# hány jó az összesből = egyeznie kell az evaluate() eredményével
print(ok/len(po.label_ids))

msg("eval")
result = trainer.evaluate()
print(f'RESULT=[{result}]')

# -----

## Hyperparameter search
#
# pip install optuna OR pip install ray[tune] is needed to do this!
#
# így kell, mert többször kell :)
#def model_init():
#    return AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)
#
#trainer = Trainer(
#    model_init=model_init,
#    args=args,
#    train_dataset=tokenized_datasets["train"],
#    eval_dataset=tokenized_datasets["validation"],
#    tokenizer=tokenizer,
#    compute_metrics=compute_metrics
#)
#
## can take a long time!!! -- try with smaller data
#best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
#
#print(best_run)
#
## To reproduce the best training,
## just set the hyperparameters in your `TrainingArgument`
## before creating a `Trainer`
#for n, v in best_run.hyperparameters.items():
#    setattr(trainer.args, n, v)
#
#trainer.train()

