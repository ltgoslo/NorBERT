#! /bin/env python3
# coding: utf-8

import pandas as pd
import torch
from torch.utils import data
from torch.optim import AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
import argparse
import logging
from datasets import ClassLabel
import tqdm
from sklearn import metrics


# This is an example of fine-tuning NorBert for the sentence classification task
# A Norwegian sentiment classification dataset is available at
# https://github.com/ltgoslo/norec_sentence/
# Example train and test files are available from https://github.com/ltgoslo/NorBERT/tree/main/utils
# (train.csv, test.csv)


def multi_acc(y_pred, y_test):
    batch_predictions = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    correctness = batch_predictions == y_test
    acc = torch.sum(correctness).item() / y_test.size(0)
    return acc, batch_predictions.tolist()


def encoder(labels, texts, cur_tokenizer, cur_device):
    labels_tensor = torch.tensor(labels).to(cur_device)
    encoding = cur_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.maxl,
    ).to(cur_device)
    return labels_tensor, encoding


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a BERT model (/cluster/shared/nlpl/data/vectors/latest/221/ "
             "or ltgoslo/norbert2 are possible options)",
        required=True,
    )
    arg(
        "--dataset",
        "-d",
        help="Path to a sentence classification train set",
        required=True,
    )
    arg(
        "--devset",
        "-dev",
        help="Path to a sentence classification dev set",
        required=True,
    )
    arg(
        "--testset",
        "-t",
        help="Path to a sentence classification test set",
        required=True,
    )
    arg("--gpu", help="Use GPU?", dest="gpu", action="store_true")
    arg("--no-gpu", help="Use GPU?", dest="gpu", action="store_false")
    arg("--epochs", "-e", type=int, help="Number of epochs", default=5)
    arg("--maxl", "-l", type=int, help="Max length", default=256)
    arg("--bsize", "-b", type=int, help="Batch size", default=16)
    arg("--save", "-s", help="Where to save the finetuned model", default="ft_bert")

    parser.set_defaults(gpu=True)
    args = parser.parse_args()

    modelname = args.model
    dataset = args.dataset
    devset = args.devset
    testset = args.testset

    logger.info("Reading train data...")
    train_data = pd.read_csv(dataset)
    logger.info("Train data reading complete.")

    logger.info("Reading dev data...")
    dev_data = pd.read_csv(devset)
    logger.info("Test data reading complete.")

    logger.info("Reading test data...")
    test_data = pd.read_csv(testset)
    logger.info("Test data reading complete.")

    num_classes = train_data["Label"].nunique()
    logger.info(f"We have {num_classes} classes")

    c2l = ClassLabel(
        num_classes=num_classes,
        names=train_data["Label"].unique().tolist(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
    model = BertForSequenceClassification.from_pretrained(modelname,
                                                          num_labels=num_classes).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_texts = train_data.Text.to_list()
    text_labels = train_data.Label.to_list()
    text_labels = [c2l.str2int(label) for label in text_labels]

    dev_texts = dev_data.Text.to_list()
    dev_labels = dev_data.Label.to_list()
    dev_labels = [c2l.str2int(label) for label in dev_labels]

    test_texts = test_data.Text.to_list()
    test_labels = test_data.Label.to_list()
    test_labels = [c2l.str2int(label) for label in test_labels]

    # We can freeze the base model and optimize only the classifier on top of it:
    freeze_model = False
    if freeze_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    logger.info(f"Tokenizing with max length {args.maxl}...")

    train_labels_tensor, train_encoding = encoder(text_labels, train_texts, tokenizer, device)
    test_labels_tensor, test_encoding = encoder(test_labels, test_texts, tokenizer, device)
    dev_labels_tensor, dev_encoding = encoder(dev_labels, dev_texts, tokenizer, device)

    input_ids = train_encoding["input_ids"]
    attention_mask = train_encoding["attention_mask"]
    dev_input_ids = dev_encoding["input_ids"]
    dev_attention_mask = dev_encoding["attention_mask"]
    test_input_ids = test_encoding["input_ids"]
    test_attention_mask = test_encoding["attention_mask"]
    logger.info("Tokenizing finished.")

    train_dataset = data.TensorDataset(input_ids, attention_mask, train_labels_tensor)
    train_iter = data.DataLoader(train_dataset, batch_size=args.bsize, shuffle=True)

    dev_dataset = data.TensorDataset(
        dev_input_ids, dev_attention_mask, dev_labels_tensor
    )
    dev_iter = data.DataLoader(dev_dataset, batch_size=args.bsize, shuffle=False)

    test_dataset = data.TensorDataset(
        test_input_ids, test_attention_mask, test_labels_tensor
    )
    test_iter = data.DataLoader(test_dataset, batch_size=args.bsize, shuffle=False)

    logger.info(f"Training with batch size {args.bsize} for {args.epochs} epochs...")

    fscores = []
    for epoch in range(args.epochs):
        losses = 0
        total_train_acc = 0
        all_predictions = []
        for text, mask, label in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            outputs = model(text, attention_mask=mask, labels=label)
            loss = outputs.loss
            losses += loss.item()
            predictions = outputs.logits
            accuracy, predicted_labels = multi_acc(predictions, label)
            all_predictions += predicted_labels
            total_train_acc += accuracy
            loss.backward()
            optimizer.step()
        train_acc = total_train_acc / len(train_iter)
        train_loss = losses / len(train_iter)

        # Testing on the dev set:
        model.eval()
        dev_predictions = []
        dev_labels = []
        for text, mask, label in tqdm.tqdm(dev_iter):
            outputs = model(text, attention_mask=mask)
            predictions = outputs.logits
            accuracy, predicted_labels = multi_acc(predictions, label)
            dev_predictions += predicted_labels
            dev_labels += label.tolist()
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(
            c2l.int2str(dev_labels),
            c2l.int2str(dev_predictions),
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
            f"Dev F1: {fscore:.4f}"
        )
        fscores.append(fscore)
        if len(fscores) > 2:
            if fscores[-1] <= fscores[-2]:
                logger.info("Early stopping!")
                break
        model.train()
    # Final testing on the test set
    predict = True

    if predict:
        model.eval()

        logger.info(f"Testing on the test set with batch size {args.bsize}...")

        test_predictions = []
        test_labels = []
        for text, mask, label in tqdm.tqdm(test_iter):
            outputs = model(text, attention_mask=mask)
            predictions = outputs.logits
            accuracy, predicted_labels = multi_acc(predictions, label)
            test_predictions += predicted_labels
            test_labels += label.tolist()
        logger.info(
            metrics.classification_report(
                c2l.int2str(test_labels), c2l.int2str(test_predictions),
                zero_division=0)
        )

        # We can try the fine-tuned model on a couple of sentences:

        sentences = [
            "Polanski er den snikende uhygges mester",
            "Utvalget diktere er skjevt .",
        ]

        for s in sentences:
            logger.info(s)
            try_encoding = tokenizer(
                [s], return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            try_encoding = try_encoding.to(device)
            input_ids = try_encoding["input_ids"]
            logger.info(tokenizer.convert_ids_to_tokens(input_ids[0]))
            attention_mask = try_encoding["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            logger.info(outputs)
            predicted = torch.log_softmax(outputs.logits, dim=1).argmax(dim=1)
            predicted_class = c2l.int2str(predicted)
            logger.info(f"This is {predicted_class[0]}")
    if args.save:
        logger.info(f"Saving the model to {args.save}...")
        model.save_pretrained(args.save)
