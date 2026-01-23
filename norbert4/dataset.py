import gzip
import os

import torch


class Dataset:

    def __init__(self, datasets: list[str], weights: list[float], tokenizer, args, seq_length, rank, is_validation=False):
        self.datasets = datasets
        self.weights = torch.tensor(weights)
        self.max_seq_length = seq_length + 1
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0
        self.selector_generator = torch.Generator().manual_seed(args.seed + args.rank - 1)

        self.mask_index = tokenizer.token_to_id(args.mask_token)
        self.cls_index = tokenizer.token_to_id(args.cls_token)
        self.pad_index = tokenizer.token_to_id(args.pad_token)

        self.doc_segments = []
        self.orders = []
        self.lens = []
        self.seed = args.seed
        if not is_validation:
            for dataset in datasets:
                if not args.train_format == "pt.gz":
                    documents = torch.load("-".join([str(dataset), f"{rank:03d}.bin"]), weights_only=False)
                else:
                    shard_path = "-".join([str(dataset), f"{rank:03d}.{args.train_format}"])
                    if not os.path.exists(shard_path):
                        shard_path = "_".join([str(dataset), f"{rank:05d}.{args.train_format}"])
                    with gzip.GzipFile(shard_path, 'rb') as f:
                        documents = torch.load(f, weights_only=False)
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
                order = RandomIndex(len(segments), args.seed + args.rank)
                self.doc_segments.append(segments)
                self.lens.append(len(segments))
                self.orders.append(order)
            self.random_indices = [
                RandomIndex(len(segments), args.seed + args.rank + 256)
                for segments in self.doc_segments
            ]

    def load_state(self, dataset_state):
        for order, state in zip(self.orders, dataset_state["orders"]):
            order.load_state(state)
        for ri, state in zip(self.random_indices, dataset_state["random_indices"]):
            ri.load_state(state)
        self.selector_generator.set_state(dataset_state["selector_generator"])

    def get_state(self):
        return {
            "orders": [order.get_state() for order in self.orders],
            "random_indices": [ri.get_state() for ri in self.random_indices],
            "selector_generator": self.selector_generator.get_state()
        }

    def set_global_step(self, global_step):
        self.global_step = global_step

    def show_random_item(self, tokenizer):
        dataset_id = torch.multinomial(self.weights, 1).item()
        index = torch.randint(0, self.lens[dataset_id], []).item()
        input_ids, target_ids, sequence_lengths, real_mask_p = self.__getitem__(dataset_id, index)
        print(' '.join(tokenizer.id_to_token(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(str(i) for i in input_ids.tolist()), flush=True)
        print()
        print(' '.join(tokenizer.id_to_token(i) if i != -100 else "-100" for i in target_ids.tolist()), flush=True)
        print()
        print(real_mask_p, flush=True)
        print()
        print(sequence_lengths, flush=True)


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments, seed):
        self.n_segments = n_segments
        self.generator = torch.Generator().manual_seed(seed)
        self.indices = torch.randperm(n_segments, generator=self.generator)
        self.index = 0
        self.counter = 0
        self.seed = seed

    def load_state(self, state):
        self.index = state["index"]
        self.counter = state["counter"]
        self.generator.manual_seed(self.seed + self.counter)
        self.indices = torch.randperm(self.n_segments, generator=self.generator)

    def get_state(self):
        return {"index": self.index, "counter": self.counter}

    def get_random_index(self):
        if self.index >= self.n_segments:
            self.counter += 1
            self.generator.manual_seed(self.seed + self.counter)
            self.indices = torch.randperm(self.n_segments, generator=self.generator)
            self.index = 0

        index = self.indices[self.index]
        self.index += 1

        return index


class MaskedDataset(Dataset):

    def __init__(self, datasets: list[str], weights: list[float], tokenizer, args, seq_length, rank, is_validation=False):
        super().__init__(datasets, weights, tokenizer, args, seq_length, rank, is_validation)

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

    def next(self, current_seq_len, batch_size):
        assert (current_seq_len*batch_size) % (self.max_seq_length-1) == 0
        assert (self.max_seq_length-1) % current_seq_len == 0
        number_of_iterations = (current_seq_len*batch_size) // (self.max_seq_length-1)
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        for _ in range(number_of_iterations):
            dataset_id = torch.multinomial(self.weights, 1, generator=self.selector_generator).item()
            index = self.orders[dataset_id].get_random_index()

            input_ids, target_ids, sequence_lengths, real_mask_p = self.__getitem__(dataset_id, index)

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)
            all_real_mask_p.append(real_mask_p)

        input_ids = torch.cat(all_input_ids)
        target_ids = torch.cat(all_target_ids)
        sequence_lengths = torch.cat(all_sequence_lengths) + torch.tensor([i // current_seq_len for i in range(current_seq_len*batch_size)])
        real_mask_p = torch.stack(all_real_mask_p).mean()

        return input_ids.unsqueeze(0), target_ids.unsqueeze(0), sequence_lengths, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / self.args.max_steps
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def __getitem__(self, dataset_id, index, increment_counts=False):
        random_index = self.random_indices[dataset_id]
        tokens = self.doc_segments[dataset_id][index]
        seq_length = min(self.max_seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()

        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            index = random_index.get_random_index()
            tokens = self.doc_segments[dataset_id][index].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            mask_ratios, replacement_tokens = self.masking_strategy(tokens)
            input_ids_, target_ids_, _ = self.apply_mask(tokens, mask_ratios, replacement_tokens)

            input_ids = torch.cat([
                input_ids,
                input_ids_,
            ])
            target_ids = torch.cat([
                target_ids,
                target_ids_
            ])

            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, real_mask_p


class CausalDataset(Dataset):

    def next(self, current_seq_len, batch_size):
        assert (current_seq_len*batch_size) % (self.max_seq_length-1) == 0
        assert (self.max_seq_length-1) % current_seq_len == 0
        number_of_iterations = (current_seq_len*batch_size) // (self.max_seq_length-1)
        all_input_ids, all_target_ids, all_sequence_lengths = [], [], []
        for _ in range(number_of_iterations):
            dataset_id = torch.multinomial(self.weights, 1, generator=self.selector_generator).item()
            index = self.orders[dataset_id].get_random_index()

            input_ids, target_ids, sequence_lengths, _ = self.__getitem__(dataset_id, index, increment_counts=True)

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)

        input_ids = torch.cat(all_input_ids)
        target_ids = torch.cat(all_target_ids)
        sequence_lengths = torch.cat(all_sequence_lengths) + torch.tensor([i // current_seq_len for i in range(current_seq_len*batch_size)])

        return input_ids.unsqueeze(0), target_ids.unsqueeze(0), sequence_lengths, torch.zeros([])

    def __getitem__(self, dataset_id, index, increment_counts=False):
        random_index = self.random_indices[dataset_id]
        tokens = self.doc_segments[dataset_id][index]
        seq_length = min(self.max_seq_length, tokens.size(0))

        input_ids = tokens[:seq_length].long()
        target_ids = tokens[:seq_length].long()

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            index = random_index.get_random_index()
            tokens = self.doc_segments[dataset_id][index].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            input_ids = torch.cat([
                input_ids,
                tokens
            ])
            target_ids = torch.cat([
                target_ids,
                tokens
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, torch.zeros([])

