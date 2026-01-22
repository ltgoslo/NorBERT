import os
import gzip

import torch

from dataset import MaskedDataset, RandomIndex

class ValidationDataset(MaskedDataset):
    def __init__(self, datasets: list[str], weights: list[float], tokenizer, args, seq_length, rank):
        super().__init__(datasets, weights, tokenizer, args, seq_length, rank, is_validation=True)

        with gzip.GzipFile(datasets[0], 'rb') as f:
            documents = torch.load(f)

        segments = []
        for i, document in enumerate(documents):
            if i % args.document_skip != 0:
                continue

            document = torch.cat([torch.LongTensor([self.cls_index]), document])
            segments += [
                document[offset : offset + self.max_seq_length]
                for offset in range(0, len(document), self.max_seq_length)
                if len(document) > 0 and len(document) - offset > 1
            ]
        n_devices = int(os.getenv("WORLD_SIZE"))
        segments = segments[:len(segments) // n_devices * n_devices]
        segments = segments[args.rank::n_devices]
        order = RandomIndex(len(segments), args.seed + args.rank)
        self.doc_segments.append(segments)
        self.lens.append(len(segments))
        self.orders.append(order)
        self.random_indices = [
            RandomIndex(len(segments), args.seed + args.rank + 256)
            for segments in self.doc_segments
        ]