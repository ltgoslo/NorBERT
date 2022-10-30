#!/bin/env python3
# coding: utf-8

# Deduplicate a set of plain-text files ending with *.gz
# Deduplication is per line
# text_dedup package is used in Python 3.9 version
# (in Python 3.8, calls are different)

import argparse
import gc
import logging
import os
import random
from itertools import repeat
from multiprocessing import Pool, Manager
from os import path
from smart_open import open
from text_dedup.near_dedup import SimHashEmbedder
from text_dedup.postprocess import get_group_indices
from text_dedup.postprocess import simhash_clustering

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_hashes(f, hasher, filenames):
    source_dir, source_file = path.split(f)
    file_index = filenames.index(source_file)
    file_hashes = {}
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
        file_hashes[computed_hash] = file_index
    data.close()
    return file_hashes


def process(f, hasher, hashes, filenames):
    source_dir, source_file = path.split(f)
    our_index = filenames.index(source_file)

    newname = "dedup_" + source_file
    outfile = path.join(source_dir, newname)
    out = open(outfile, "a")
    data = open(f)
    logger.info(f"De-duplicating {f}...")
    # logger.info(f"Our index is {our_index}...")

    written_hashes = set()
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

        #  Checking if this line occurred in other files:
        indices = hashes[computed_hash]
        preceding = [ind for ind in indices if ind < our_index]

        # Only saving the line if it was not seen before in the current file
        # AND we are the first file to encounter it:
        if len(preceding) == 0 and computed_hash not in written_hashes:
            written_hashes.add(computed_hash)
            out.write(line + "\n")
        # Else discarding the line as a duplicate:
        else:
            discarded += 1
            if len(examples) > 10:
                examples.remove(random.sample(list(examples), 1)[0])
            examples.add(line)
    data.close()
    out.close()
    return total, discarded, short, examples


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

    datafilenames = [path.split(n)[1] for n in datafiles]
    logger.info(f"Calculating hashes of {corpus}...")

    manager = Manager()

    paralellism_hash = 32 if len(datafiles) > 32 else len(datafiles)
    paralellism_dedup = 32 if len(datafiles) > 32 else len(datafiles)

    with Pool(paralellism_hash) as p:
        computed_hashes = p.starmap(compute_hashes, zip(datafiles, repeat(embedder),
                                                        repeat(datafilenames)))
    logger.info(f"Computing hashes complete.")
    keys = set().union(*computed_hashes)
    logger.info(f"Merging hashes complete.")
    embeddings = {el: set() for el in keys}
    for dic in computed_hashes:
        for el in dic:
            embeddings[el].add(dic[el])
    embeddings = manager.dict(embeddings)
    logger.info(f"{len(embeddings)} unique hashes in total")

    del computed_hashes
    gc.collect()

    with Pool(paralellism_dedup) as p:
        results = p.starmap(process, zip(datafiles, repeat(embedder),
                                         [embeddings for i in range(len(datafiles))],
                                         repeat(datafilenames)))

    all_total = sum([el[0] for el in results])
    all_discarded = sum([el[1] for el in results])
    all_short = sum([el[2] for el in results])
    all_examples = [el[3] for el in results]
    all_examples = set().union(*all_examples)

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
