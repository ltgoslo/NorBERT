# /bin/env python3
# coding: utf-8

import sys
import stanza

#stanza.download("no")
#stanza.download("nn")

lang = sys.argv[1]  # State the segmenter model (no or nn)

nlp = stanza.Pipeline(lang, processors="tokenize", tokenize_batch_size=256)

stack = []

counter = 0

for line in sys.stdin:
    doc = nlp(line.strip())
    for sentence in doc.sentences:
        if len(sentence.text) > 2:
            print(sentence.text)
    print("")
    counter += 1
    if counter % 10000 == 0:
        print(f"{counter} lines processed", file=sys.stderr)

