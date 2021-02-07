# /bin/env python3
# coding: utf-8

import argparse
import collections
import os
import time
import numpy as np
from simple_elmo import ElmoModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    LSTM,
    Bidirectional,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import logging
from sklearn.metrics import classification_report
import random as python_random
import pandas as pd


def infer_sentence_embeddings(texts, contextualized, layers="average"):
    start = time.time()
    sentence_elmo_vectors = contextualized.get_elmo_vector_average(texts, layers=layers)
    end = time.time()
    processing_time = int(end - start)
    logger.info(
        f"ELMo embeddings for your input are ready in {processing_time} seconds"
    )
    if layers == "all":
        sentence_elmo_vectors = sentence_elmo_vectors.reshape(
            sentence_elmo_vectors.shape[0],
            sentence_elmo_vectors.shape[1] * sentence_elmo_vectors.shape[2],
        )
    logger.info(f"Tensor shape: {sentence_elmo_vectors.shape}")
    return sentence_elmo_vectors


def infer_token_embeddings(texts, contextualized, layers="average"):
    start = time.time()
    elmo_vectors = contextualized.get_elmo_vectors(texts, layers=layers)
    end = time.time()
    processing_time = int(end - start)
    logger.info(
        f"ELMo embeddings for your input are ready in {processing_time} seconds"
    )
    if layers == "all":
        elmo_vectors = elmo_vectors.reshape(
            elmo_vectors.shape[0],
            elmo_vectors.shape[2],
            elmo_vectors.shape[1] * elmo_vectors.shape[3],
        )

    logger.info(f"Tensor shape: {elmo_vectors.shape}")
    return elmo_vectors


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--input", "-i", help="Path to a directory with CSV data", required=True)
    arg("--elmo", "-e", help="Path to ELMo model", required=True)
    arg(
        "--method", "-m", help="How to classify", choices=["bow", "lstm"], default="bow"
    )
    arg(
        "--elmo_layers",
        "-l",
        help="What ELMo layers to use?",
        default="average",
        choices=["average", "all", "top"],
    )

    args = parser.parse_args()
    data_path = args.input
    method = args.method
    elmo_layers = args.elmo_layers
    elmoname = args.elmo.strip().split("/")[-2]

    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    trainfile = None
    devfile = None
    testfile = None
    for file in os.scandir(data_path):
        if file.name.endswith(".csv"):
            if "train" in file.name:
                trainfile = file
            elif "dev" in file.name:
                devfile = file
            elif "test" in file.name:
                testfile = file

    if trainfile is None or devfile is None or testfile is None:
        raise SystemExit("Not all necessary CSV files found!")

    logger.info("Loading the datasets...")
    train_data = pd.read_csv(trainfile, header=None)
    dev_data = pd.read_csv(devfile, header=None)
    test_data = pd.read_csv(testfile, header=None)
    logger.info("Finished loading the datasets")

    y_train = train_data[0].values
    y_dev = dev_data[0].values
    y_test = test_data[0].values

    classes = sorted(list(set(y_train)))
    num_classes = len(classes)
    logger.info(f"{num_classes} classes")
    counter = collections.Counter(y_train)

    logger.info("===========================")
    logger.info("Class distribution in the training data:")
    for el in counter:
        logger.info(f"{el}\t{counter[el]}")
    logger.info("===========================")

    # Converting text labels to indexes
    y_train = [classes.index(i) for i in y_train]
    y_dev = [classes.index(i) for i in y_dev]
    y_test = [classes.index(i) for i in y_test]

    # Convert indexes to binary class matrix (for use with categorical_crossentropy loss)
    y_train = to_categorical(y_train, num_classes)
    logger.info(f"Train labels shape: {y_train.shape}")
    y_dev = to_categorical(y_dev, num_classes)
    y_test = to_categorical(y_test, num_classes)

    el_model = ElmoModel()
    el_model.load(args.elmo, max_batch_size=64)

    x_sentences = [s.split() for s in train_data[1].values]
    x_dev_sentences = [s.split() for s in dev_data[1].values]
    x_test_sentences = [s.split() for s in test_data[1].values]
    logger.info(f"{len(x_sentences)} train texts")
    average_length = np.mean([len(t) for t in x_sentences])
    max_length = np.max([len(t) for t in x_sentences])
    logger.info(f"Average train text length: {average_length:.3f} words")
    logger.info(f"Max train text length: {max_length:.3f} words")

    MAXLEN = min(50, max_length)
    if method == "lstm":
        logger.info(f"Chosen max sentence length for LSTM: {MAXLEN}")

    logger.info("Inferring embeddings for train...")
    if method == "bow":
        x_train = infer_sentence_embeddings(x_sentences, el_model, layers=elmo_layers)
        logger.info("Inferring embeddings for dev...")
        x_dev = infer_sentence_embeddings(x_dev_sentences, el_model, layers=elmo_layers)
    else:
        x_train = infer_token_embeddings(
            [s[:MAXLEN] for s in x_sentences], el_model, layers=elmo_layers
        )
        logger.info("Inferring embeddings for dev...")
        x_dev = infer_token_embeddings(
            [s[:MAXLEN] for s in x_dev_sentences], el_model, layers=elmo_layers
        )

    # Basic type of TensorFlow models: a sequential stack of layers
    model = Sequential()

    if method == "bow":
        # The first layer maps the ELMo representations into low-dimensional hidden representations:
        model.add(
            Dense(
                128, input_shape=(x_train.shape[1],), activation="relu", name="Input",
            )
        )
        model.add(Dropout(0.1))  # We will use dropout after the first hidden layer
    else:
        model.add(
            Bidirectional(
                LSTM(16, return_sequences=True, recurrent_dropout=0.1),
                input_shape=(MAXLEN, x_train.shape[2]),
            )
        )
        model.add(GlobalMaxPooling1D())

    # model.add(Dense(128, activation="relu"))

    model.add(Dense(num_classes, activation="softmax", name="Output"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Print out the model architecture
    logger.info(model.summary())

    # We will monitor the dynamics of accuracy on the validation set during training
    # If it stops improving, we will stop training.
    earlystopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.0001, patience=3, verbose=1, mode="max"
    )

    # Train the compiled model on the training data
    # See more at https://keras.io/models/sequential/#sequential-model-methods
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        verbose=2,
        validation_data=(x_dev, y_dev),
        batch_size=32,
        callbacks=[earlystopping],
    )

    logger.info("Inferring embeddings for test...")
    if method == "bow":
        x_test = infer_sentence_embeddings(
            x_test_sentences, el_model, layers=elmo_layers
        )
    else:
        x_test = infer_token_embeddings(
            [s[:MAXLEN] for s in x_test_sentences], el_model, layers=elmo_layers
        )

    score = model.evaluate(x_test, y_test, verbose=2)
    logger.info(f"Test loss: {score[0]:.4f}")
    logger.info(f"Test accuracy: {score[1]:.4f}")

    # We use the sklearn classification_report() function
    # to calculate per-class F1 score on the dev set:
    predictions = model.predict(x_test)
    # map predictions to the binary {0, 1} range:
    predictions = np.around(predictions)

    # Convert predictions from integers back to text labels:
    y_test_real = [classes[int(np.argmax(pred))] for pred in y_test]
    predictions = [classes[int(np.argmax(pred))] for pred in predictions]

    train_accuracies = history.history["accuracy"]
    dev_accuracies = history.history["val_accuracy"]
    epochs_series = range(len(train_accuracies))

    df = pd.DataFrame(
        list(zip(epochs_series, train_accuracies, dev_accuracies)),
        columns=["epoch", "train_score", "dev_score"],
    )
    df.to_csv(f"{elmoname}_sentiment_{method}_log.tsv", sep="\t", index=False)

    cls_rep = classification_report(y_test_real, predictions)
    logger.info("Classification report for the test set:")
    logger.info(cls_rep)
    with open(f"{elmoname}_sentiment_{method}_report.txt", "w") as f:
        f.write(cls_rep)
