# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized

from tokenizers import Tokenizer
import json
import os
import argparse
# from smart_open import open
import torch
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subcorpus', type=str, required=True)
    return parser.parse_args()


def tokenize(tokenizer, text):
    text = text.rstrip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int32)

    return ids


if __name__ == "__main__":
    args = parse_args()

    # load the tokenizer
    tokenizer = Tokenizer.from_file("norsk_data/tokenizer.json")

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])

    input_dir = Path("norsk_data/shards")
    subcorpus = args.subcorpus
    output_dir = Path("norsk_data/tokenized_shards")

    input_filename = input_dir / f"{subcorpus}-{rank:03d}.jsonl"
    output_filename = output_dir / f"{subcorpus}-{rank:03d}.bin"

    # tokenize file
    tokenized_documents = []
    n_subwords = 0
    for i, line in enumerate(tqdm(open(input_filename, 'rt'), desc=f"Tokenizing {input_filename}", disable=rank != 0)):
        document = json.loads(line)["text"]
        tokenized_document = tokenize(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

        if i == 0 and rank == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.decode([token]))
            print(flush=True)

    # save the tokenized documents
    torch.save(tokenized_documents, output_filename)

    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")
