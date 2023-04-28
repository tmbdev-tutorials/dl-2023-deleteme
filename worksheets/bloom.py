#!/usr/bin/env python
# coding: utf-8

import sys
from transformers import BloomTokenizerFast, BloomForTokenClassification, BloomForCausalLM
from transformers import pipeline, TextGenerationPipeline
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mname = "bigscience/bloom-560m"
mname = "bigscience/bloom-1b1"
mname = "bigscience/bloom-1b7"
mname = "bigscience/bloom-7b1"
mname = "bigscience/bloom"
class bloom_model:
    def __init__(self):
        self.tokenizer = BloomTokenizerFast.from_pretrained(mname)
        self.model = BloomForCausalLM.from_pretrained(mname).to(device)

        self.generator = TextGenerationPipeline(
            task = 'text-generation',
            model = self.model,
            tokenizer = self.tokenizer,
            device = device,
        )

    def predict(self, content: str) -> str:
            answer = self.generator(content, max_new_tokens=100)
            return answer

print("loading", mname, file=sys.stderr)
model = bloom_model()
print("done")

while True:
    line = input("? ")
    if line == "exit":
        break
    output = model.predict(line.strip())
    print(output[0]['generated_text'])
