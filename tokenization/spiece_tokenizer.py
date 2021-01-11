#! /bin/env python3

import sys
from tokenizers import SentencePieceBPETokenizer

tokenizer = SentencePieceBPETokenizer()

tokenizer.train([sys.argv[1]], vocab_size=30000)

tokenizer.save(sys.argv[2] + ".json")
