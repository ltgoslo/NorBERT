import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=32)
    parser.add_argument("--data_dir", type=str, default="norsk_data")
    parser.add_argument("--subcorpus", type=str, default="all_se.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.data_dir = Path(args.data_dir)

    target_dir = args.data_dir / "shards"
    if not target_dir.exists():
        target_dir.mkdir()
    
    source_path = args.data_dir / args.subcorpus
    target_paths = [target_dir / f"{args.subcorpus[:-6]}-{i:03d}.jsonl" for i in range(args.n_shards)]
    target_files = [open(target_path, "w") for target_path in target_paths]

    for i, line in enumerate(open(source_path)):
        line = json.loads(line)
        if isinstance(line, str):
            text = line.strip()
        else:
            text = line["text"].strip()

        target_files[i % args.n_shards].write(json.dumps({"text": text}) + "\n")

    for target_file in target_files:
        target_file.close()