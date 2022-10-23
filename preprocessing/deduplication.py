#!/bin/env python3
# coding: utf-8

# Deduplicate a set of plain-text files ending with *.gz
# Deduplication is per line

import argparse
from smart_open import open
import logging
from text_dedup.near_dedup import SimHashEmbedder
from text_dedup.postprocess import simhash_clustering
from text_dedup.postprocess import get_group_indices
import os
from os import path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", "-c", help="Path to the corpus (can be compressed)", required=True)
    arg("--logname", "-l", help="Name of the log file", default="corpus_deduplication")
    arg("--mode", "-m", help="Identical or near-deduplication", choices=["identical", "near"],
        default="identical")
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

    if args.mode == "near":
        embeddings = []
    else:
        embeddings = set()

    datafiles = []
    corpus = args.corpus
    if path.isfile(corpus):
        datafiles.append(corpus)
    elif path.isdir(corpus):
        datafiles = [path.join(corpus, f) for f in os.listdir(corpus)
                     if path.isfile(path.join(corpus, f)) and f.endswith(".gz")]
    logger.info(f"Calculating hashes of {corpus}...")

    for f in datafiles:
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
            if args.mode == "near":
                embeddings.append(computed_hash)
            else:
                if computed_hash in embeddings:
                    discarded += 1
                    continue
                embeddings.add(computed_hash)
                print(line)
    logger.info(f"Processing {total} lines complete, {len(embeddings)} unique hashes in total")
    if args.mode == "identical":
        logger.info(f"{discarded} duplicate lines discarded, {short} short lines left as is.")
        exit()

    if args.mode == "near":
        logger.info(f"Clustering hashes...")
        clusters = simhash_clustering(embeddings)
        groups = get_group_indices(clusters)
        logger.info(f"Clustering done")
        for el in groups:
            print(el)
