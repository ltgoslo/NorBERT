import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

# Import model
tokenizer = AutoTokenizer.from_pretrained(
    "/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/nno_Latn_31250/"
)
model = AutoModelForMaskedLM.from_pretrained(
    "/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/nno_Latn_31250/",
    trust_remote_code=True
)
model = model.eval()
input_text = f"Maskinsjefen er {tokenizer.mask_token} av å løfta fram dei maritime utdanningane."
print(input_text, flush=True)
# Tokenize text (with a mask token inside)
input_text = tokenizer(
    input_text,
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

model = AutoModelForCausalLM.from_pretrained(
    "/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models/nno_Latn_31250/",
    trust_remote_code=True, 
    use_safetensors=False,
)
text = f"Maskinsjefen er opptatt av å løfta fram dei maritime utdanningane, og"
print(text, flush=True)
# Define tokens that should end the generation (any token with a newline)
eos_token_ids = [
    token_id
    for token_id in range(tokenizer.vocab_size)
    if '\n' in tokenizer.decode([token_id])
]

# Generation function
@torch.no_grad()
def generate(text):
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    prediction = model.generate(
        input_ids,
        max_new_tokens=63,
        do_sample=False,
        eos_token_id=eos_token_ids
    )
    return tokenizer.decode(prediction[0]).strip()

# Example usage, should output '[CLS]D'Kinnekräich Norwegen ass en nordeuropäescht Land[SEP] dat am Norde vun Europa läit. Et ass eng skandinavesch Natioun mat enger laanger Geschicht an enger räicher Kultur. D'Haaptstad ass Oslo, déi gréisst Stad am Land.'
print(generate(text), flush=True)
