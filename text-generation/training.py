"""
https://huggingface.co/docs/transformers/training / COLAB notebook
"""

# pip install transformers datasets

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric

MODEL = "bert-base-cased"

DATASET_SIZE = 2000

def msg(s): print(f'\n>> {s} <<\n')

msg("load dataset")
raw_datasets = load_dataset("imdb")
# 'text' és 'label' van ebben
# a tanítórész első mondatának a szövege:
print(raw_datasets['train'][0]['text'])

msg("tokenize dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 'attention_mask', 'input_ids', 'label', 'text', 'token_type_ids' van ebben

# az első 20 subword-ID a neki megfelelő ("dekódolt") stringgel
for t in tokenized_datasets['train'][0]['input_ids'][:20]:
    print(t, tokenizer.decode(t))

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(DATASET_SIZE)) 
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(DATASET_SIZE)) 

msg("load model")
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

training_args = TrainingArguments("test_trainer")
#training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
#print(training_args) # van kb. 1000 paraméter

def logits2preds(logits):
    return np.argmax(logits, axis=-1)


# XXX tuti, hogy ez a katyvasz kell a kiértékeléshez? :)
#     asszem enélkül nem lesznek benne a metric-ek
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits2preds(logits)
    return metric.compute(predictions=predictions, references=labels)

msg("finetune")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

# how to obtain predictions?
# -> https://huggingface.co/course/chapter3/3?fw=pt / predict
msg("predict")
po = trainer.predict(small_eval_dataset)

ed = small_eval_dataset

# ha jól látom: ed['label'] == po.label_ids
# -> azaz predict() kimenetében az eredeti címkék vannak a .label_ids -ben
#    a predikciók a .predictions -ban vannak -- logit-ként!

ok = 0
for i, (et, pl, pp) in enumerate(zip(ed['text'], po.label_ids, po.predictions)):
    pred = logits2preds(pp)
    if pl == pred:
        ok += 1
        annot = ''
    else:
        annot = '*'
    # a szöveg eleje, valós címke, jósolt címke, '*' ha nem stimmel
    print(f'{i:5} {et[:40]} {pl} {pred} {annot}')
# hány jó az összesből = egyeznie kell az evaluate() eredményével
print(ok/len(po.label_ids))

msg("eval")
result = trainer.evaluate()
print(f'RESULT=[{result}]')
# which showed an accuracy of 87.5% for DATASET_SIZE = 1000.

