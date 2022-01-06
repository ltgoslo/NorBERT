from transformers import BertForMaskedLM, AutoTokenizer, BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import sys


tokenizer = BertTokenizer.from_pretrained(sys.argv[1], use_fast=False)
model = BertForMaskedLM.from_pretrained(sys.argv[1])

text = "Jeg vi elsker dette landet."
TOKEN_INDEX = 3

encoded_input = tokenizer(text, return_tensors="pt")["input_ids"]
print(tokenizer.decode(encoded_input[0, :]))

MASK_ID = tokenizer.convert_tokens_to_ids("[MASK]")

encoded_input[0, TOKEN_INDEX] = MASK_ID
print(tokenizer.decode(encoded_input[0, :]))

output = model(input_ids=encoded_input)["logits"][0, TOKEN_INDEX, :]

print(
    f"ARGMAX AT INDEX {TOKEN_INDEX}: {tokenizer.convert_ids_to_tokens([output.argmax()])}"
)

top = np.argpartition(output.detach().numpy(), -4)[-4:]

print("Top four:", tokenizer.convert_ids_to_tokens(top))

_, ax = plt.subplots(1, 1)
ax.hist(output.detach().numpy(), bins=29)
plt.show()
