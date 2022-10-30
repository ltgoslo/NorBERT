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
from multiprocessing import Pool, Manager
from itertools import repeat
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_hashes(f, hasher):
    file_hashes = set()
    data = open(f)
    logger.info(f"Computing hashes from {f}...")
    for line in data:
        line = line.strip()
        # We do not include very short lines in deduplication:
        if len(line.split()) < 5:
            continue
        # Blank lines are also discarded
        if not line:
            continue
        computed_hash = hasher.embed_function()(line)
        file_hashes.add(computed_hash)
    data.close()
    return file_hashes


def process(f, hasher, hashes, marker):
    source_dir, source_file = path.split(f)

    newname = marker + "_" + source_file
    outfile = path.join(source_dir, newname)
    out = open(outfile, "a")
    data = open(f)
    logger.info(f"De-duplicating {f}...")

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

        if computed_hash in hashes:
            discarded += 1
            if len(examples) > 10:
                examples.remove(random.sample(list(examples), 1)[0])
            examples.add(line)
            continue
        else:
            out.write(line + "\n")
    data.close()
    out.close()
    return total, discarded, short, examples


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

    manager = Manager()
    paralellism_hash = 32 if len(datafiles1) > 32 else len(datafiles1)
    paralellism_dedup = 16 if len(datafiles2) > 16 else len(datafiles2)

    logger.info(f"Computing hashes from the reference corpus {corpus1}...")
    with Pool(paralellism_hash) as p:
        computed_hashes = p.starmap(compute_hashes, zip(datafiles1, repeat(embedder)))
    logger.info(f"Computing hashes complete.")
    embeddings = set().union(*computed_hashes)
    embeddings = manager.dict(embeddings)
    logger.info(f"Processing complete, {len(embeddings)} unique reference hashes in total")

    del computed_hashes
    gc.collect()

    logger.info(f"De-deduplicating corpus {corpus2}...")

    with Pool(paralellism_dedup) as p:
        results = p.starmap(process, zip(datafiles2, repeat(embedder),
                                         [embeddings for i in range(len(datafiles2))],
                                         repeat(args.logname)))

    all_total = sum([el[0] for el in results])
    all_discarded = sum([el[1] for el in results])
    all_short = sum([el[2] for el in results])
    all_examples = [el[3] for el in results]
    all_examples = set().union(*all_examples)

    logger.info(f"{all_total} lines processed.")
    logger.info(f"{all_discarded} duplicate lines discarded ("
                f"{(all_discarded / all_total) * 100:.3f}%), "
                f"{all_short} short lines left as is.")
    logger.info("Some examples of discarded lines:")
    for el in list(all_examples)[:11]:
        logger.info(f"'{el}'")

