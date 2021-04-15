import tqdm
import torch
import logging
from torch.nn import functional
from torch import nn
from torch.utils.data import DataLoader
from dataset import CoNLLDataset
from nermodel import NERmodel
from transformers import BertTokenizer
from smart_open import open
import argparse
from conllu import parse
from functools import partial

def build_mask(tokenizer, ids):
    tok_sents = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    mask = []
    for sentence in tok_sents:
        current = []
        for n, token in enumerate(sentence):
            if token in tokenizer.all_special_tokens[1:] or token.startswith("##"):
                continue
            else:
                current.append(n)
        mask.append(current)

    mask = tokenizer.pad({"input_ids": mask}, return_tensors="pt")["input_ids"]
    return mask


def predict(input_data, tokenizer, model, gpu=False):
    input_data = tokenizer(
        input_data, is_split_into_words=True, return_tensors="pt", padding=True
    )
    if gpu:
        input_data = input_data.to("cuda")
    batch_mask = build_mask(tokenizer, input_data["input_ids"])
    y_pred = model(input_data, batch_mask).permute(0, 2, 1).argmax(dim=1)
    return y_pred


def collate_fn(batch, gpu=False):
    longest_y = max([y.size(0) for X, y in batch])
    x = [X for X, y in batch]
    y = torch.stack(
        [functional.pad(y, (0, longest_y - y.size(0)), value=-1) for X, y in batch])
    if gpu:
        y = y.to("cuda")
    return x, y


def load_data(filename):
    entries = []
    with open(filename, "r") as f:
        current = []
        for line in f:
            if line.startswith("#"):
                continue

            if not line.rstrip():
                entries.append(current)
                current = []
                continue

            res = line.strip().replace("–", "-").split("\t")
            current.append(res)

    return entries


def main():
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--train",
        "-t",
        help="Path to the training dataset",
        required=True,
        default="norne-nb-in5550-train.conllu.gz",
    )
    arg(
        "--dev",
        "-d",
        help="Path to the development dataset",
        required=True,
        default="norne-nb-in5550-dev.conllu.gz",
    )
    arg("--test", help="Path to the test dataset")
    arg(
        "--bert",
        "-b",
        help="Path to BERT model",
        default="/cluster/shared/nlpl/data/vectors/latest/216/",
    )
    arg("--epoch", "-e", help="Number of epochs to train", default=20, type=int)
    arg(
        "--name",
        "-n",
        help="Name of the file to save predictions",
        default="predictions.conllu.gz",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu = True
    else:
        gpu = False

    # Do we want to freeze BERT itself and only train a classifier on top of it?
    freeze = False

    logger.info(args.bert)
    logger.info(args.train)
    train_data = load_data(args.train)
    val_data = load_data(args.dev)
    train_data = CoNLLDataset(train_data)
    dev_data = CoNLLDataset(val_data, ner_vocab=train_data.ner_vocab)

    train_loader = DataLoader(
        train_data, batch_size=32, shuffle=True, collate_fn=partial(collate_fn, gpu=gpu)
    )
    dev_loader = DataLoader(
        dev_data, batch_size=16, shuffle=False, collate_fn=partial(collate_fn, gpu=gpu)
    )
    logger.info("Data loaded")

    model_path = args.bert
    if gpu:
        model = NERmodel(train_data.ner_vocab, model_path=model_path, freeze=freeze).to("cuda")
    else:
        model = NERmodel(train_data.ner_vocab, model_path=model_path, freeze=freeze)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if freeze:
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=2e-5)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_basic_tokenize=False)

    current_acc = 0.0
    # train/eval loop
    for epoch in range(args.epoch):
        model.train()
        logger.info(f"Epoch {epoch}")
        train_iter = tqdm.tqdm(train_loader)
        for x, y in train_iter:
            x = tokenizer(
                x, is_split_into_words=True, return_tensors="pt", padding=True
            )
            if gpu:
                x = x.to("cuda")
            batch_mask = build_mask(tokenizer, x["input_ids"])
            optimiser.zero_grad()
            y_pred = model(x, batch_mask).permute(0, 2, 1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            train_iter.set_postfix_str(f"loss: {loss.item()}")

        model.eval()
        dev_iter = tqdm.tqdm(dev_loader)
        no_ne_index = train_data.ner_indexer["name=O"]
        correct, total = 0, 0
        for x, y in dev_iter:
            y_pred = predict(x, tokenizer, model, gpu=gpu)
            correct += (
                ((y_pred == y).logical_and(y != no_ne_index))
                .nonzero(as_tuple=False)
                .size(0)
            )
            total += (
                (y != no_ne_index).logical_and(y != -1).nonzero(as_tuple=False).size(0)
            )

        accuracy = correct / total
        logger.info(f"Validation accuracy = {correct} / {total} = {accuracy}")
        if accuracy < current_acc:
            logger.info(f"Early stopping at epoch {epoch}!")
            break
        else:
            current_acc = accuracy

    if args.test:
        predicted_tags = []
        test_conll = parse(open(args.test, "r").read())
        test_data = CoNLLDataset(load_data(args.test), ner_vocab=train_data.ner_vocab)

        test_loader = DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=partial(collate_fn, gpu=gpu)
        )
        logger.info("Test data loaded")

        test_iter = tqdm.tqdm(test_loader)
        for x, y in test_iter:
            y_pred = predict(x, tokenizer, model, gpu=gpu)
            predicted = [train_data.ner_vocab[el].split("=")[-1] for el in y_pred[0]]
            predicted_tags += predicted

        counter = 0
        for nr, sentence in enumerate(test_conll):
            for token in sentence:
                token["misc"]["name"] = predicted_tags[counter]
                counter += 1

        with open(args.name, "w") as f:
            for sentence in test_conll:
                f.write(sentence.serialize())
        logger.info(f"Predictions written to {args.name}")


if __name__ == "__main__":
    main()
