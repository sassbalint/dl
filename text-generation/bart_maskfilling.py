"""
https://huggingface.co/docs/transformers/model_doc/bart
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

import sys

# "This is a very <mask> book." -> good
# "This is a very <mask> boy." -> happy
# "I like to read <mask> and newspapers." -> books, magazines,
# "I like skiing, <mask> and skating." -> snowboarding

TEXT = "UN Chief Says There Is No <mask> in Syria"
if len(sys.argv) > 1: TEXT = sys.argv[1]

MODEL = "facebook/bart-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BartForConditionalGeneration.from_pretrained(MODEL, forced_bos_token_id=0)
model = model.to(device)

tokenizer = BartTokenizer.from_pretrained(MODEL)

batch = tokenizer(TEXT, return_tensors='pt')
batch = batch.to(device)

generated_ids = model.generate(batch['input_ids'])

generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_sentence)

