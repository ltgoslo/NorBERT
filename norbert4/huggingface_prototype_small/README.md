---
language:
- ltg
inference: false
tags:
- BERT
- HPLT
- encoder
- text2text-generation
license: apache-2.0
datasets:
- HPLT/HPLT3.0
---

# HPLT v3.0 GPT-BERT for Latgalian

<img src="https://hplt-project.org/_next/static/media/logo-hplt.d5e16ca5.svg" width=12.5%>

This is one of the monolingual language models trained as a third release by the [HPLT project](https://hplt-project.org/).
Our models follow the setup of [GPT-BERT](https://aclanthology.org/2024.conll-babylm.24/). 

- hidden size: 384
- attention heads: 6
- layers: 12
- vocabulary size: 32768

Every model uses its own tokenizer trained on language-specific HPLT data. 

[The training code](https://github.com/ltgoslo/NorBERT/tree/main/norbert4).

## Example usage (bidirectional encoding)

This model currently needs a custom wrapper from `modeling_gptbert.py`, you should therefore load the model with `trust_remote_code=True`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import model
tokenizer = AutoTokenizer.from_pretrained(
    "HPLT/hplt_gpt_bert_small_3_0_ltg_Latn",
)
model = AutoModelForMaskedLM.from_pretrained(
    "HPLT/hplt_gpt_bert_small_3_0_ltg_Latn",
    trust_remote_code=True,
    use_safetensors=False,
)
model = model.eval()
input_text = f"Norwegian is a {tokenizer.mask_token} Germanic language"
print(input_text)
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

# Decoding; should output: 'Norwegian is a North Germanic language'
print(tokenizer.decode(output_text[0].tolist()))
```

## Example usage (text generation)

GPT-BERT also supports unidirectional text decoding, it can generate text like any other GPT model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "HPLT/hplt_gpt_bert_small_3_0_ltg_Latn",
)
model = AutoModelForCausalLM.from_pretrained(
    "HPLT/hplt_gpt_bert_small_3_0_ltg_Latn",
    trust_remote_code=True, 
    use_safetensors=False,
)
text = f"The Norwegian Constitution"
print(text, flush=True)
# Define tokens that should end the generation
eos_token_ids = [
    token_id
    for token_id in range(tokenizer.vocab_size)
    if '.' in tokenizer.decode([token_id])
]

# Generation function
@torch.no_grad()
def generate(text):
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    prediction = model.generate(
        input_ids,
        max_new_tokens=63,
        do_sample=False,
        eos_token_id=eos_token_ids,
    )
    return tokenizer.decode(prediction[0]).strip()

# Example usage, should output '[CLS]The Norwegian Constitution[SEP]is a document that defines the rights and responsibilities of the Norwegian people and their representatives.'
print(generate(text), flush=True)
```

The following classes are currently implemented: `AutoModel`, `AutoModelForMaskedLM`, `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` and `AutoModeltForMultipleChoice`.

## Intermediate checkpoints

We are releasing 10 intermediate checkpoints for each model at intervals of every 3125 training steps in separate branches. The naming convention is `stepXXX`: for example, `step18750`.

You can load a specific model revision with `transformers` using the argument `revision`:
```python
model = AutoModelForSeq2SeqLM.from_pretrained("HPLT/hplt_gpt_bert_small_3_0_ltg_Latn", revision="step21875", trust_remote_code=True)
```

You can access all the revisions for the models with the following code:
```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("HPLT/hplt_gpt_bert_small_3_0_ltg_Latn")
print([b.name for b in out.branches])
```

## Cite us

```bibtex
@inproceedings{charpentier-samuel-2024-bert,
    title = "{GPT} or {BERT}: why not both?",
    author = "Charpentier, Lucas Georges Gabriel  and
      Samuel, David",
    booktitle = "The 2nd BabyLM Challenge at the 28th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2024",
    address = "Miami, FL, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.conll-babylm.24/",
    pages = "262--283"
}
```

```bibtex
@misc{oepen2025hplt30largescalemultilingual,
      title={{HPLT 3.0}: {V}ery Large-Scale Multilingual Resources for {LLM} and {MT}. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models}, 
      author={Stephan Oepen and Nikolay Arefev and Mikko Aulamo and Marta Bañón and Maja Buljan and Laurie Burchell and Lucas Charpentier and Pinzhen Chen and Mariia Fedorova and Ona de Gibert and Barry Haddow and Jan Hajič and Jindřich Helcl and Andrey Kutuzov and Veronika Laippala and Zihao Li and Risto Luukkonen and Bhavitvya Malik and Vladislav Mikhailov and Amanda Myntti and Dayyán O'Brien and Lucie Poláková and Sampo Pyysalo and Gema Ramírez Sánchez and Janine Siewert and Pavel Stepachev and Jörg Tiedemann and Teemu Vahtola and Dušan Variš and Fedor Vitiugin and Tea Vojtěchová and Jaume Zaragoza},
      year={2025},
      eprint={2511.01066},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01066}, 
}
```
[![arXiv](https://img.shields.io/badge/arXiv-2410.24159-b31b1b.svg)](https://arxiv.org/abs/2410.24159)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01066-b31b1b.svg)](https://arxiv.org/abs/2511.01066)
