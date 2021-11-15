#!/bin/env python3
# coding: utf-8

import sys

for line in sys.stdin:
    if not line:
        continue
    #if len(line.strip()) == 1:
    #    continue
    text = line.encode("utf-8", "ignore")
    out = text.decode("utf-8", "ignore").strip()
    print(out)

