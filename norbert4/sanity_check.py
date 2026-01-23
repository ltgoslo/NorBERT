import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import model
tokenizer = AutoTokenizer.from_pretrained(
    "/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/deu_Latn_9375/"
)
model = AutoModelForMaskedLM.from_pretrained(
    "/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/deu_Latn_9375/",
    trust_remote_code=True
)
model = model.eval()

# Tokenize text (with a mask token inside)
input_text = tokenizer(
    f"Ich suche eine {tokenizer.mask_token} Wohnung.",
    return_tensors="pt",
)
# Inference
with torch.no_grad():
    output_p = model(**input_text)

# Unmask the text
output_text = torch.where(
    input_text.input_ids == tokenizer.mask_token_id,
    output_p.logits.argmax(-1),
    input_text.input_ids
)

# Decoding; should output: '<s>Nå ønsker de seg en ny bolig.'
print(tokenizer.decode(output_text[0].tolist()), flush=True)
