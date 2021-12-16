"""
https://huggingface.co/eugenesiow/bart-paraphrase
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

import sys

TEXT = "They were there to enjoy us and they were there to pray for us."
if len(sys.argv) > 1: TEXT = sys.argv[1]

MODEL = "eugenesiow/bart-paraphrase"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForConditionalGeneration.from_pretrained(MODEL)
model = model.to(device)

tokenizer = BartTokenizer.from_pretrained(MODEL)

batch = tokenizer(TEXT, return_tensors='pt')
batch = batch.to(device)

generated_ids = model.generate(batch['input_ids'])

generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_sentence)

