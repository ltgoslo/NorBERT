#!/bin/env python3
# coding: utf-8

# Deduplicate a set of plain-text files ending with *.gz
# Deduplication is per line
# text_dedup package is used in Python 3.9 version
# (in Python 3.8, calls are different)

import argparse
from smart_open import open
import logging
from text_dedup.near_dedup import SimHashEmbedder
from text_dedup.postprocess import simhash_clustering
from text_dedup.postprocess import get_group_indices
import os
from os import path
import random


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def process(f, hasher, hashes):
    source_dir, source_file = path.split(f)
    newname = "dedup_" + source_file
    outfile = path.join(source_dir, newname)
    out = open(outfile, "a")
    data = open(f)
    logger.info(f"De-duplicating {f}...")

    seen_hashes = set()
    total = 0
    discarded = 0
    short = 0
    examples = set()

    for line in data:
        line = line.strip()
        # We do not include very short lines in deduplication:
        if len(line.split()) < 5:
            short += 1
            total += 1
            out.write(line + "\n")
            continue
        # Blank lines are also simply printed as they are
        if not line:
            out.write("\n")
            continue
        computed_hash = hasher.embed_function()(line)
        total += 1
        # Only saving the line if it was not seen before
        if computed_hash not in hashes and computed_hash not in seen_hashes:
            seen_hashes.add(computed_hash)
            out.write(line + "\n")
        # Else discarding the line as a duplicate:
        else:
            discarded += 1
            if len(examples) > 10:
                examples.remove(random.sample(list(examples), 1)[0])
            examples.add(line)
    data.close()
    out.close()
    return total, discarded, short, examples, seen_hashes


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
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO,
                        handlers=[
                            logging.FileHandler(logfile),
                            logging.StreamHandler()
                        ]
                        )

    logger = logging.getLogger(__name__)

    embedder = SimHashEmbedder()

    datafiles = []
    corpus = args.corpus
    if path.isfile(corpus):
        datafiles.append(corpus)
    elif path.isdir(corpus):
        datafiles = [path.join(corpus, f) for f in os.listdir(corpus)
                     if path.isfile(path.join(corpus, f)) and f.endswith(".gz")]

    logger.info(f"Deduplicating {corpus}...")

    embeddings = set()
    all_total = 0
    all_discarded = 0
    all_short = 0
    all_examples = set()

    for datafile in datafiles:
        cur_total, cur_discarded, cur_short, cur_examples, cur_embeddings = process(
            datafile, embedder, embeddings)
        all_total += cur_total
        all_discarded += cur_discarded
        all_short += cur_short
        all_examples |= cur_examples
        embeddings |= cur_embeddings

    logger.info(f"Computing hashes complete.")
    logger.info(f"{len(embeddings)} unique hashes in total")

    logger.info(f"{all_total} lines processed.")
    if args.mode == "identical":
        logger.info(f"{all_discarded} duplicate lines discarded ("
                    f"{(all_discarded / all_total) * 100:.3f}%), "
                    f"{all_short} short lines left as is.")
        logger.info("Some examples of discarded lines:")
        for el in list(all_examples)[:11]:
            logger.info(f"'{el}'")
        exit()

    if args.mode == "near":
        logger.info(f"Clustering hashes...")
        clusters = simhash_clustering(embeddings)
        groups = get_group_indices(clusters)
        logger.info(f"Clustering done")
        for el in groups:
            print(el)
