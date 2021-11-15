# /bin/env python3
# coding: utf-8

import sys
import stanza

# stanza.download("no")
# stanza.download("nn")

lang = sys.argv[1]  # State the segmenter model (no or nn)

nlp = stanza.Pipeline(lang, processors="tokenize")

stack = []

for line in sys.stdin:
    doc = nlp(line.strip())
    for sentence in doc.sentences:
        if len(sentence.text) > 2:
            print(sentence.text)
