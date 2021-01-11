#!/bin/env python3
# coding: utf-8

# Convert from one-token-per-line to plain text

import sys

sentence_end_markers = {".", "!", "?", "¶"}

for line in sys.stdin:
    line = line.strip()
    if line == "|":
        continue
    if line.startswith("##") and line.endswith(">"):
        if line.startswith("##U"):
            print()
        continue
    if line.endswith("¶"):
        line = line.replace("¶", "").strip()
    if len(line.split()) > 1:
        print(line)


