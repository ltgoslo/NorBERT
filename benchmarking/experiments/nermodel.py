from transformers import BertModel
from torch import nn


class NERmodel(nn.Module):
    def __init__(self, ner_vocab, model_path=None, freeze=False):
        super().__init__()
        self._bert = BertModel.from_pretrained(
            model_path if model_path else "bert-base-multilingual-cased"
        )
        hidden_size = self._bert.config.hidden_size
        self._linear = nn.Linear(hidden_size, len(ner_vocab))

        if freeze:
            for param in self._bert.parameters():
                param.requires_grad = False

    def forward(self, batch, mask):
        b = self._bert(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        pooler = b.last_hidden_state[:, mask].diagonal().permute(2, 0, 1)
        return self._linear(pooler)
