#!/bin/env python3

from transformers import TFBertForTokenClassification
from data_preparation.data_preparation_pos import MBERTTokenizer as MBERT_Tokenizer_pos
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        modelname = sys.argv[1]
    else:
        modelname = "ltgoslo/norbert"
    model = TFBertForTokenClassification.from_pretrained(modelname, from_pt=True)
    tokenizer = MBERT_Tokenizer_pos.from_pretrained(modelname, do_lower_case=False)

