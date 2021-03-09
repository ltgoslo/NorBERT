#! /bin/env python3
# coding: utf-8

import pandas as pd
import torch
from torch.utils import data
from transformers import AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
import argparse

# This is an example of fine-tuning NorBert for the sentence classification task
# A Norwegian sentiment classification dataset is available at
# https://github.com/ltgoslo/NorBERT/tree/main/benchmarking/data/sentiment/no


def multi_acc(y_pred, y_test):
    batch_predictions = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    correctness = batch_predictions == y_test
    acc = torch.sum(correctness).item() / y_test.size(0)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a BERT model (/cluster/shared/nlpl/data/vectors/latest/216/ "
        "or ltgoslo/norbert are possible options)",
        required=True,
    )
    arg("--dataset", "-d", help="Path to a document classification dataset", required=True)
    arg("--gpu", "-g", help="Use GPU?", type=bool, default=True)
    arg("--epochs", "-e", type=int, help="Number of epochs", default=10)

    args = parser.parse_args()
    modelname = args.model
    dataset = args.dataset

    tokenizer = AutoTokenizer.from_pretrained(modelname)
    if args.gpu:
        model = BertForSequenceClassification.from_pretrained(modelname).to("cuda")
    else:
        model = BertForSequenceClassification.from_pretrained(modelname)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    print("Reading train data...")
    train_data = pd.read_csv(dataset)
    train_data.columns = ["labels", "text"]
    print("Train data reading complete.")

    texts = train_data.text.to_list()
    text_labels = train_data.labels.to_list()

    # We can freeze the base model and optimize only the classifier on top of it:
    freeze_model = False
    if freeze_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    print("Tokenizing...")
    if args.gpu:
        labels = torch.tensor(text_labels).to("cuda")
        encoding = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to("cuda")
    else:
        labels = torch.tensor(text_labels)
        encoding = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    print("Tokenizing finished.")

    train_dataset = data.TensorDataset(input_ids, attention_mask, labels)
    train_iter = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(args.epochs):
        losses = 0
        total_train_acc = 0
        for i, (text, mask, label) in enumerate(train_iter):
            optimizer.zero_grad()
            outputs = model(text, attention_mask=mask, labels=label)
            loss = outputs.loss
            losses += loss.item()
            predictions = outputs.logits
            accuracy = multi_acc(predictions, label)
            total_train_acc += accuracy
            loss.backward()
            optimizer.step()
        train_acc = total_train_acc / len(train_iter)
        train_loss = losses / len(train_iter)
        print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # We can try the fine-tuned model on a couple of sentences:
    predict = False

    if predict:
        model.eval()

        sentences = [
            "Polanski er den snikende uhygges mester",
            "Utvalget diktere er skjevt .",
        ]

        for s in sentences:
            print(s)
            encoding = tokenizer(
                [s], return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            input_ids = encoding["input_ids"]
            print(tokenizer.convert_ids_to_tokens(input_ids[0]))
            attention_mask = encoding["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            print(outputs)
