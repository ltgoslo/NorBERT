#!/bin/env python3
# coding: utf-8

# Text extraction from NAK XML files

import sys
from xml.dom import minidom

file = sys.argv[1]

print(file, file=sys.stderr)

doc = minidom.parse(file)

divs = doc.getElementsByTagName("div")

valid = {"title", "caption", "text"}

for d in divs:
    if d.getAttributeNode("type").nodeValue in valid:
        paragraphs = d.getElementsByTagName("p")
        if paragraphs:
            par_data = []
            for paragraph in paragraphs:
                if paragraph.hasChildNodes():
                    text = paragraph.childNodes[-1]
                    if text.hasChildNodes():
                        text = text.childNodes[-1]
                    par_data.append(text.data.replace('\n', ' '))
            output = '\n'.join(par_data)
        else:
            if d.hasChildNodes():
                output = d.childNodes[-1].data.strip().replace('\n', ' ')
            else:
                output = None
        if output:
            print(output.strip())
print()
