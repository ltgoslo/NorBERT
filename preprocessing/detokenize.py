#!/bin/env python3
# coding: utf-8

# Convert from one-token-per-line to plain text

import sys
from nltk.tokenize.treebank import TreebankWordDetokenizer

sentence_end_markers = {".", "!", "?", "¶"}

stack = []

for line in sys.stdin:
    line = line.strip()
    if line == "|":
        continue
    if line.startswith("<") and line.endswith(">"):
        if line.startswith("<U"):
            if stack:
                text = TreebankWordDetokenizer().detokenize(stack)
                print(text)
                stack = []
            print()
        continue
    stack.append(line)
    if line in sentence_end_markers:
        text = TreebankWordDetokenizer().detokenize(stack)
        print(text)
        stack = []


