#! /bin/env python3

import sys
from tokenizers import SentencePieceBPETokenizer
import gensim

tokenizer = SentencePieceBPETokenizer()

data = gensim.models.word2vec.LineSentence(sys.argv[1])

tokenizer.train_from_iterator(data, vocab_size=30000)

tokenizer.save(sys.argv[2] + ".json")
