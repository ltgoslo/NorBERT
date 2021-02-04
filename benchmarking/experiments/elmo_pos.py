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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import logging
from sklearn.metrics import classification_report
from utils.utils import read_conll
import random as python_random
import pandas as pd


def infer_embeddings(texts, contextualized):
    start = time.time()
    elmo_vectors = contextualized.get_elmo_vectors(texts, layers="average")
    end = time.time()
    processing_time = int(end - start)
    logger.info(f"ELMo embeddings for your input are ready in {processing_time} seconds")
    logger.info(f"Tensor shape: {elmo_vectors.shape}")

    nr_words = len([item for sublist in texts for item in sublist])

    feature_matrix = np.zeros((nr_words, contextualized.vector_size))
    nr = 0
    for vect, sent in zip(elmo_vectors, texts):
        cropped_matrix = vect[: len(sent), :]
        for row in cropped_matrix:
            feature_matrix[nr] = row
            nr += 1
    return feature_matrix


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--input", "-i", help="Path to a directory with CONLLu files", required=True
    )
    arg("--elmo", "-e", help="Path to ELMo model", required=True)

    args = parser.parse_args()
    data_path = args.input

    lang = data_path.strip().split("/")[-2]
    elmoname = args.elmo.strip().split("/")[-2]

    trainfile = None
    devfile = None
    testfile = None
    for file in os.scandir(data_path):
        if file.name.endswith(".conllu"):
            if "-train" in file.name:
                trainfile = file
            elif "-dev" in file.name:
                devfile = file
            elif "-test" in file.name:
                testfile = file

    if trainfile is None or devfile is None or testfile is None:
        raise SystemExit("Not all necessary CONLL files found!")

    logger.info("Loading the datasets...")
    train_data = read_conll(trainfile)
    dev_data = read_conll(devfile)
    test_data = read_conll(testfile)
    logger.info("Finished loading the datasets")

    y_train = [item for sublist in train_data[2] for item in sublist]
    y_dev = [item for sublist in dev_data[2] for item in sublist]
    y_test = [item for sublist in test_data[2] for item in sublist]

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

    x_sentences = train_data[1]
    x_dev_sentences = dev_data[1]
    x_test_sentences = test_data[1]
    logger.info(f"{len(x_sentences)} train texts")
    average_length = np.mean([len(t) for t in x_sentences])
    logger.info(f"Average train text length: {average_length:.3f} words")

    logger.info("Inferring embeddings for train and dev...")
    x_train = infer_embeddings(x_sentences, el_model)
    x_dev = infer_embeddings(x_dev_sentences, el_model)

    # Basic type of TensorFlow models: a sequential stack of layers
    model = (Sequential())

    # We now start adding layers.
    # The first layer maps the ELMo representations into low-dimensional hidden representations:
    model.add(
        Dense(128,
              input_shape=(el_model.vector_size,),
              activation="relu",
              name="Input", ))

    model.add(Dropout(0.1))  # We will use dropout after the first hidden layer

    # model.add(Dense(128, activation="relu"))

    model.add(
        Dense(num_classes, activation="softmax", name="Output"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

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
        epochs=20,
        verbose=2,
        validation_data=(x_dev, y_dev),
        batch_size=64,
    #   callbacks=[earlystopping],
    )

    logger.info("Inferring embeddings for test...")
    x_test = infer_embeddings(x_test_sentences, el_model)

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

    df = pd.DataFrame(list(zip(epochs_series, train_accuracies, dev_accuracies)), columns =["epoch", "train_score", "dev_score"])
    df.to_csv(f"{lang}_{elmoname}_pos_log.tsv", sep="\t", index=False)

    logger.info("Classification report for the test set:")
    logger.info(classification_report(y_test_real, predictions))

