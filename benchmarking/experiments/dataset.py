import torch
from torch.utils.data import Dataset


class CoNLLDataset(Dataset):
    def __init__(self, entries, ner_vocab=None):
        self.forms = [[current[1] for current in entry] for entry in entries]
        self.ner = [[current[-1].split("|")[-1] for current in entry] for entry in entries]

        if ner_vocab:
            self.ner_vocab = ner_vocab
        else:
            self.ner_vocab = list(set([item for sublist in self.ner for item in sublist]))
            self.ner_vocab.extend(['@UNK'])
        self.ner_indexer = {i: n for n, i in enumerate(self.ner_vocab)}

    def __getitem__(self, index):
        forms = self.forms[index]
        ner = self.ner[index]

        x = forms
        y = torch.LongTensor([self.ner_indexer[i] if i in self.ner_vocab
                              else self.ner_indexer['@UNK'] for i in ner])
        return x, y

    def __len__(self):
        return len(self.forms)
