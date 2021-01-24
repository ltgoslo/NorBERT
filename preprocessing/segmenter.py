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

    if line == "\n":
        texts = "\n".join(stack)
        doc = nlp(texts)
        for sentence in doc.sentences:
            print(sentence.text)
        stack = []
        print("")
        continue
    stack.append(line.strip())
