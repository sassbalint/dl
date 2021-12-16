"""
https://huggingface.co/docs/transformers/model_doc/encoderdecoder
"""

import torch
from transformers import EncoderDecoderModel, BertTokenizer

import sys

ARTICLE = "This is a really long text"
SUMMARY = "This is the corresponding summary"

MODEL = "bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL, MODEL)
# initialize Bert2Bert from pre-trained checkpoints
model = model.to(device)

tokenizer = BertTokenizer.from_pretrained(MODEL)

# hack with special tokens -- see bertgeneration.py
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
# XXX add_cross_attention=True ??? is_decoder=True ???
# -> lehet, hogy ezek benne vannak a
#    from_encoder_decoder_pretrained()-ben

input_ids = tokenizer(ARTICLE, return_tensors="pt").input_ids
input_ids = input_ids.to(device)
labels = tokenizer(SUMMARY, return_tensors="pt").input_ids
labels = labels.to(device)

# training
outputs = model(input_ids=input_ids, labels=input_ids)
loss, logits = outputs.loss, outputs.logits

# generation
generated = model.generate(input_ids)

print(generated)

