"""
https://huggingface.co/docs/transformers/model_doc/bertgeneration
"""

import torch
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer

import sys

ARTICLE = "This is a long article to summarize"
SUMMARY = "This is a short summary"

MODEL = "bert-large-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hack: use BERT's [CLS] token as BOS token and [SEP] token as EOS token
encoder = BertGenerationEncoder.from_pretrained(MODEL, bos_token_id=101, eos_token_id=102)
# main point: "add cross attention layers" (??)
decoder = BertGenerationDecoder.from_pretrained(MODEL, add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102)
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
model = model.to(device)

tokenizer = BertTokenizer.from_pretrained(MODEL)

input_ids = tokenizer(ARTICLE, add_special_tokens=False, return_tensors="pt").input_ids
input_ids = input_ids.to(device)
labels = tokenizer(SUMMARY, return_tensors="pt").input_ids
labels = labels.to(device)

# training
outputs = model(input_ids=input_ids, decoder_input_ids=labels, labels=labels)
loss, logits = outputs.loss, outputs.logits
loss.backward() # XXX ???

print(f'LOSS=[{loss}]')

# XXX nem vili, hogy mi történik...

