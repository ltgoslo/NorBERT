#!/bin/env python3
# coding: utf-8

# Strip the tab symbols

import sys

for line in sys.stdin:
    if not line:
        print()
        continue
    if len(line.strip()) == 1:
        continue
    out = line.replace("¶", "").strip()
    print(out)

