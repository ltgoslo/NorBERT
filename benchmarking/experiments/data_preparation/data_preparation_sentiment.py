#!/bin/env python3

import tensorflow as tf
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers.data.processors.utils import InputFeatures


class Example:
    def __init__(self, text, category_index):
        self.text = text
        self.category_index = category_index


def bert_convert_examples_to_tf_dataset(
        examples,
        tokenizer,
        max_length=64,
):
    """
    Loads data into a tf.data.Dataset for finetuning a given model.

    Args:
        examples: List of tuples representing the examples to be fed
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum string length

    Returns:
        a ``tf.data.Dataset`` containing the condensed features of the provided sentences
    """
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default
            truncation=True
        )

        # input ids = token indices in the tokenizer's internal dict
        # token_type_ids = binary mask identifying different sequences in the model
        # attention_mask = binary mask indicating the positions of padded tokens
        # so the model does not attend to them

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"],
                                                     input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                label=e.category_index
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


def roberta_convert_examples_to_tf_dataset(
        examples,
        tokenizer,
        max_length=64,
):
    """
    Loads data into a tf.data.Dataset for finetuning a given model.

    Args:
        examples: List of tuples representing the examples to be fed
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum string length

    Returns:
        a ``tf.data.Dataset`` containing the condensed features of the provided sentences
    """
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default
            truncation=True
        )

        # input ids = token indices in the tokenizer's internal dict
        # token_type_ids = binary mask identifying different sequences in the model
        # attention_mask = binary mask indicating the positions of padded tokens
        # so the model does not attend to them

        input_ids, attention_mask = (input_dict["input_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, label=e.category_index
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


def load_dataset(lang_path, tokenizer, max_length, balanced=False,
                 dataset_name="test", limit=None):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tqdm.pandas(leave=False)
    # Read data
    df = pd.read_csv(lang_path + "/{}.csv".format(dataset_name.split("_")[0]), header=None)
    df.columns = ["sentiment", "review"]
    df["sentiment"] = pd.to_numeric(df["sentiment"])  # Sometimes label gets read as string

    # Remove excessively long examples
    lengths = df["review"].progress_apply(lambda x: len(tokenizer.encode(x)))
    df = df[lengths <= max_length].reset_index(drop=True)  # Remove long examples

    # Balance classes
    if dataset_name == "train" and balanced:
        positive_examples = df["sentiment"].sum()
        if not limit:
            # Find which class is the minority and set its size as limit
            n = min(positive_examples, df.shape[0] - positive_examples)
        else:
            n = limit
        ones_idx = np.random.choice(np.where(df["sentiment"])[0], size=n)
        zeros_idx = np.random.choice(np.where(df["sentiment"] == 0)[0], size=n)
        df = df.loc[list(ones_idx) + list(zeros_idx)].reset_index(drop=True)
    elif not balanced and limit:
        raise Exception("Must set 'balanced' to True to choose a manual limit.")

    # Convert to TF dataset
    dataset = bert_convert_examples_to_tf_dataset(
        [(Example(text=text, category_index=label)) for label, text in df.values], tokenizer,
        max_length=max_length)
    return df, dataset
