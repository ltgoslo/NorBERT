#!/bin/env python3
# coding: utf-8

# Deduplicate a plain-text file (per line)

import argparse
from smart_open import open
import logging
from text_dedup.embedders import SimHashEmbedder
from text_dedup.postprocess.clustering import simhash_clustering
from text_dedup.postprocess.group import get_group_indices
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    # corpus = [
    #     "The quick brown fox jumps over the lazy dog",
    #     "The quick brown fox jumps over the lazy dog",
    #     "This is a test",
    #     "This is a test",
    # ]
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--corpus", help="Path to the corpus (can be compressed)", required=True)
    arg("--mode", "-m", help="Identical or near-deduplication", choices=["identical", "near"],
        default="identical")
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    embedder = SimHashEmbedder()

    corpus = args.corpus
    data = open(corpus)

    discarded = 0
    short = 0
    logger.info(f"Calculating hashes of {corpus}...")
    if args.mode == "near":
        embeddings = []
    else:
        embeddings = set()
    for line in data:
        line = line.strip()
        # We do not include very short lines in deduplication:
        if len(line.split()) < 5:
            short += 1
            print(line)
        computed_hash = embedder.embed_function()(line)
        if args.mode == "near":
            embeddings.append(computed_hash)
        else:
            if computed_hash in embeddings:
                discarded += 1
                continue
            embeddings.add(computed_hash)
            print(line)
    logger.info(f"Calculating {len(embeddings)} hashes complete")
    if args.mode == "identical":
        logger.info(f"{discarded} duplicate lines discarded, {short} short lines left as is.")

    if args.mode == "near":
        logger.info(f"Clustering hashes...")
        clusters = simhash_clustering(embeddings)
        groups = get_group_indices(clusters)
        logger.info(f"Clustering done")
        for el in groups:
            print(el)
