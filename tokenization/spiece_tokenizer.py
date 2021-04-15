#! /bin/env python3

import sys
from tokenizers import SentencePieceBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

import gensim

tokenizer = SentencePieceBPETokenizer()
pre_tokenizer = Whitespace()
tokenizer.pre_tokenizer = pre_tokenizer

data = gensim.models.word2vec.LineSentence(sys.argv[1])

tokenizer.train_from_iterator(data, vocab_size=30000)

tokenizer.save(sys.argv[2] + ".json")
