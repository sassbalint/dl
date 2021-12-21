"""
Finetune for token classification.
"""

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
import numpy as np

# https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb
MODEL = "distilbert-base-uncased"
TASK = ("conll2003",)
# keys: 'id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'
# note: "tokens" (= list of strs) instead of "sentence" (str) here!
DATACOLUMN= "tokens" # ~ words
LABELCOLUMN = "ner_tags" # "ner_tags", "pos_tags" or "chunk_tags"
# new keys after tok: 'attention_mask', 'input_ids' + 'labels'!
TRAIN = "train"
EVAL = "validation"
METRIC = ("seqeval",)
SAVE_DIR = f'{MODEL}-finetuned-{"-".join(TASK)}-{LABELCOLUMN}'

DATASET_SIZE = 0 # 0 means all data
BATCH_SIZE = 32 # based on gpustat -cupF

EXAMPLE = 4 # index of example taken from data
EXAMPLES = EXAMPLE + 1 # to define range 0..EXAMPLES


def section(s): print(f'\n>> {s} <<\n')
def msg(msg, obj, inline=False):
    sep = ': ' if inline else '\n'
    print(f'** {msg}{sep}{obj}\n')

section("load dataset")
raw_datasets = load_dataset(*TASK)
msg('raw datasets', raw_datasets)
msg(f'raw/train/{EXAMPLE}', raw_datasets[TRAIN][EXAMPLE])
LABEL_LIST = raw_datasets[TRAIN].features[LABELCOLUMN].feature.names
msg(f'raw/train/features/{LABELCOLUMN}',
    raw_datasets[TRAIN].features[LABELCOLUMN])

section("tokenize dataset")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
# You can check whether the used model have a FastTokenizer here:
# https://huggingface.co/transformers/index.html#bigtable

example = raw_datasets[TRAIN][EXAMPLE]
tokenized_example = tokenizer(example[DATACOLUMN], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_example["input_ids"])
msg('# ex', len(example[DATACOLUMN]), inline=True)
msg('ex', list(zip(example[DATACOLUMN], example[LABELCOLUMN])))
msg('# ex/subwords', len(tokens), inline=True)
msg('ex/subwords', tokens)

section("tokenize dataset / align labels")
# TokenClassification megy, így aztán
# meg kell oldani, hogy minden subword-höz átírjuk
# az eredetileg a szónál meglévő label-eket!
#
# word_ids() (= map subwords to words) alapján szépen megoldható :)
word_ids = tokenized_example.word_ids()
IGNORED = -100 # index ignored by PyTorch
# XXX ez igaziból a tokenize_and_aling_labels() egy sorban! :)
# XXX minek a másik? :)
aligned_labels = [IGNORED if i is None else example[LABELCOLUMN][i] for i in word_ids]
msg('labels', example[LABELCOLUMN])
msg('word_ids', word_ids)
msg('aligned_labels', aligned_labels)
msg('ex/subwords/aligned', list(zip(tokens, aligned_labels)))

# két módszer van a címketerjeszésre
# pl. ha van egy egy-subword-ös szó + egy 3-subword-ös szó:
#  [1] 1, 2, 2, 2
#  [2] 1, 2, IGNORED, IGNORED
# az alábbi flag-gel ez állítható:
label_all_tokens = True

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples[DATACOLUMN],
        truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[LABELCOLUMN]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Spec tokens' (subwords'...) word_id is None.
            # Set the label to IGNORED
            # to be ignored in the loss function.
            if word_idx is None:
                label_ids.append(IGNORED)
            # 1st subword of word: set the label.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # other subwords: set the label or set it to IGNORED.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else IGNORED)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenize_function = tokenize_and_align_labels

msg('tokenized', tokenize_function(raw_datasets[TRAIN][:EXAMPLES]))

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# az első 20 subword-ID a neki megfelelő ("dekódolt") stringgel
for t in tokenized_datasets[TRAIN][EXAMPLE]['input_ids'][:20]:
    print(t, tokenizer.decode(t))

msg('tokenized datasets', tokenized_datasets)

if DATASET_SIZE > 0:
    train_dataset = tokenized_datasets[TRAIN].shuffle(seed=42).select(range(DATASET_SIZE))
    eval_dataset = tokenized_datasets[EVAL].shuffle(seed=42).select(range(DATASET_SIZE))
else:
    train_dataset = tokenized_datasets[TRAIN]
    eval_dataset = tokenized_datasets[EVAL]

section("load model")
num_labels = len(LABEL_LIST)
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=num_labels)

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

    # a hf access token is needed to do this
    #push_to_hub=True,
)

# appropriate padding, etc. (?)
section("data collator")
data_collator = DataCollatorForTokenClassification(tokenizer)

section("metric")
metric = load_metric(*METRIC)

fake_labels = [[LABEL_LIST[i] for i in example[LABELCOLUMN]]] # not exactly fake :)
fake_preds = [['O' for i in example[LABELCOLUMN]]]
msg('fake labels', fake_labels)
msg('fake preds', fake_preds)
msg(f'example `{METRIC}` metric calculation', metric.compute(predictions=fake_preds, references=fake_labels))

def logits2preds(logits):
    return np.argmax(logits, axis=-1) # last dimension

# XXX why do we need to convert to string label?
# 3-dim labels+preds: sentence, token, value <- TokenClassification
def compute_metrics(eval_pred, full_result=False):
    predicted_logits, pre_labels = eval_pred
    pre_predictions = logits2preds(predicted_logits)
    # ignore IGNORED special tokens
    # XXX how to solve by map+filter / list comprehension? :)
    predictions, labels = [], []
    for prediction, label in zip(pre_predictions, pre_labels):
        ps, ls = [], []
        for (p, l) in zip(prediction, label):
            if l != IGNORED:
                ps.append(LABEL_LIST[p])
                ls.append(LABEL_LIST[l])
        predictions.append(ps)
        labels.append(ls)
    results = metric.compute(predictions=predictions, references=labels)
    if not full_result:
        # keep only "overall" measures
        results = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return results

section("finetune")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# get P/R/F1 for each category at the end of training
section("predict")
preds, labels, _ = trainer.predict(eval_dataset)
eval_pred = (preds, labels)
results = compute_metrics(eval_pred, full_result=True)
msg('prediction results', results)

section("eval")
result = trainer.evaluate()
msg('RESULT', result)

