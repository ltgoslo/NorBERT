#!/bin/env python3
# coding: utf-8

# Deduplicate one corpus based on another
# Deduplication is per line
# text_dedup package is used in Python 3.9 version
# (in Python 3.8, calls are different)

import argparse
from smart_open import open
import logging
from text_dedup.near_dedup import SimHashEmbedder
import os
from os import path
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus1", "-c1", help="Path to the reference corpus (file or directory)", required=True)
    arg("--corpus2", "-c2", help="Path to the corpus to de-dupe)", required=True)
    arg("--logname", "-l", help="Name of the log file", default="corpus_deduplication")
    args = parser.parse_args()

    logfile = args.logname + "_dedup.log"
    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    embedder = SimHashEmbedder()

    total = 0
    discarded = 0
    short = 0

    embeddings = set()
    examples = set()

    datafiles1 = []
    corpus1 = args.corpus1
    if path.isfile(corpus1):
        datafiles1.append(corpus1)
    elif path.isdir(corpus1):
        datafiles1 = [path.join(corpus1, f) for f in os.listdir(corpus1)
                      if path.isfile(path.join(corpus1, f)) and f.endswith(".gz")]

    datafiles2 = []
    corpus2 = args.corpus2
    if path.isfile(corpus2):
        datafiles2.append(corpus2)
    elif path.isdir(corpus2):
        datafiles2 = [path.join(corpus2, f) for f in os.listdir(corpus2)
                      if path.isfile(path.join(corpus2, f)) and f.endswith(".gz")]

    logger.info(f"Computing hashes from the reference corpus {corpus1}...")
    for f in datafiles1:
        data = open(f)
        logger.info(f"Processing {f}...")
        for line in data:
            line = line.strip()
            # We do not include very short lines in deduplication:
            if len(line.split()) < 5:
                continue
            # Blank lines are also skipped
            if not line:
                continue
            computed_hash = embedder.embed_function()(line)
            embeddings.add(computed_hash)
    logger.info(f"Processing complete, {len(embeddings)} unique reference hashes in total")

    logger.info(f"De-deduplicating corpus {corpus2}...")
    for f in datafiles2:
        data = open(f)
        logger.info(f"Processing {f}...")
        for line in data:
            line = line.strip()
            # We do not include very short lines in deduplication:
            if len(line.split()) < 5:
                short += 1
                total += 1
                print(line)
                continue
            # Blank lines are also simply printed as they are
            if not line:
                print()
                continue
            computed_hash = embedder.embed_function()(line)
            total += 1
            if computed_hash in embeddings:
                discarded += 1
                if len(examples) > 10:
                    examples.remove(random.sample(list(examples), 1)[0])
                examples.add(line)
                continue
            print(line)
    logger.info(f"Processing {total} lines complete.")
    logger.info(f"{discarded} duplicate lines discarded, {short} short lines left as is.")
    logger.info("Some examples of discarded sequences:")
    for el in examples:
        logger.info(el)
