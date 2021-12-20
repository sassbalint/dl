"""
https://huggingface.co/docs/transformers/training / COLAB notebook
"""

# pip install transformers datasets

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

MODEL = "bert-base-cased"
DATASET_SIZE = 100
#DATASET_SIZE = 2000
BATCH_SIZE = 4 # based on `gpustat -cp --watch`

def msg(s): print(f'\n>> {s} <<\n')

msg("load dataset")
raw_datasets = load_dataset("imdb") # 'text', 'label'
DATACOLUMN = "text"
# a tanítórész első mondatának a szövege:
print(raw_datasets)
print(raw_datasets['train'][0])

msg("tokenize dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL) # use_fast=True

def tokenize_function(examples):
    return tokenizer(examples[DATACOLUMN], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 'attention_mask', 'input_ids', 'label', 'text', 'token_type_ids' van ebben

# az első 20 subword-ID a neki megfelelő ("dekódolt") stringgel
for t in tokenized_datasets['train'][0]['input_ids'][:20]:
    print(t, tokenizer.decode(t))

# ha kisebb kell: random sample
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(DATASET_SIZE)) 
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(DATASET_SIZE)) 

msg("load model")
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

args = TrainingArguments(
    "test_trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=5,
)
# van még kb. 1000 paraméter...

msg("metric")
metric = load_metric("accuracy")

def logits2preds(logits):
    return np.argmax(logits, axis=-1) # last dimension

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
    tokenizer=tokenizer, # XXX was not here...
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
# which showed an accuracy of 87.5% for DATASET_SIZE = 1000.

