#! /bin/env python3
# ! coding: utf-8

import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import AdamW
import pandas as pd

modelname = "ltgoslo/norbert"

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = BertForSequenceClassification.from_pretrained(modelname)
model.to('cuda')
model.train()

optimizer = AdamW(model.parameters(), lr=1e-5)

train_data = pd.read_csv("train.csv")
train_data.columns = ["labels", "text"]

# We will use the first 10 instances for this simple example:
text_batch = train_data.text.to_list()[:10]
text_labels = train_data.labels.to_list()[:10]

encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

labels = torch.tensor(text_labels).unsqueeze(0).to('cuda')

for epoch in range(20):
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss}")
