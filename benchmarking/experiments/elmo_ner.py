# /bin/env python3
# coding: utf-8

import argparse
import collections
import logging
import random as python_random
import time
import numpy as np
import tensorflow as tf
from conllu import parse
from simple_elmo import ElmoModel
from smart_open import open
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from elmodel import keras_ner


def infer_embeddings(texts, contextualized, layers="average"):
    start = time.time()
    elmo_vectors = contextualized.get_elmo_vectors(texts, layers=layers)

    nr_words = len([item for sublist in texts for item in sublist])

    feature_matrix = np.zeros((nr_words, contextualized.vector_size))
    row_nr = 0
    for vect, sent in zip(elmo_vectors, texts):
        cropped_matrix = vect[: len(sent), :]
        for row in cropped_matrix:
            feature_matrix[row_nr] = row
            row_nr += 1

    end = time.time()
    processing_time = int(end - start)

    logger.info(
        f"ELMo embeddings for your input are ready in {processing_time} seconds"
    )
    logger.info(f"Tensor shape: {feature_matrix.shape}")

    return feature_matrix


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    # For reproducibility:
    np.random.seed(42)
    python_random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--train", "-t", help="Path to the train CONLLU file", required=True)
    arg("--dev", "-d", help="Path to the dev CONLLU file", required=True)
    arg("--elmo", "-e", help="Path to ELMo model", required=True)
    arg("--test", help="Path to the test CONLLU file")
    arg("--name", "-n", help="Name for the CONLLU file to save", default="predictions.conllu.gz")
    arg(
        "--elmo_layers",
        "-l",
        help="What ELMo layers to use?",
        default="average",
        choices=["average", "all", "top"],
    )

    args = parser.parse_args()
    elmo_layers = args.elmo_layers

    elmoname = args.elmo.strip().split("/")[-1]

    logger.info("Loading the datasets...")
    # Returns  texts, tags
    train_conll = parse(open(args.train, "r").read())
    dev_conll = parse(open(args.dev, "r").read())

    train_data = [[token["form"] for token in sentence] for sentence in train_conll], \
                 [[token["misc"]["name"] for token in sentence] for sentence in train_conll]
    dev_data = [[token["form"] for token in sentence] for sentence in dev_conll], \
               [[token["misc"]["name"] for token in sentence] for sentence in dev_conll]
    logger.info("Finished loading the datasets")

    y_train = [item for sublist in train_data[1] for item in sublist]
    y_dev = [item for sublist in dev_data[1] for item in sublist]

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

    # Convert indexes to binary class matrix (for use with categorical_crossentropy loss)
    y_train = to_categorical(y_train, num_classes)
    logger.info(f"Train labels shape: {y_train.shape}")
    y_dev = to_categorical(y_dev, num_classes)

    el_model = ElmoModel()
    el_model.load(args.elmo, max_batch_size=64)

    x_sentences = train_data[0]
    x_dev_sentences = dev_data[0]
    logger.info(f"{len(x_sentences)} train texts")
    average_length = np.mean([len(t) for t in x_sentences])
    logger.info(f"Average train text length: {average_length:.3f} words")

    counter = 0
    for nr, sentence in enumerate(dev_conll):
        for token in sentence:
            token["misc"]["name"] = "B_PER"
            counter += 1

    logger.info("Inferring embeddings for train and dev...")
    x_train = infer_embeddings(x_sentences, el_model, layers=elmo_layers)
    x_dev = infer_embeddings(x_dev_sentences, el_model, layers=elmo_layers)

    model = keras_ner(input_shape=el_model.vector_size, hidden_size=128, num_classes=num_classes)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Print out the model architecture
    logger.info(model.summary())

    # We will monitor the dynamics of accuracy on the validation set during training
    # If it stops improving, we will stop training.
    earlystopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.0001, patience=2, verbose=1, mode="max"
    )

    # Train the compiled model on the training data
    # See more at https://keras.io/models/sequential/#sequential-model-methods
    history = model.fit(
        x_train,
        y_train,
        epochs=20,
        verbose=2,
        validation_data=(x_dev, y_dev),
        batch_size=32,
        callbacks=[earlystopping],
    )
    if args.test:
        test_conll = parse(open(args.test, "r").read())

        test_data = [[token["form"] for token in sentence] for sentence in test_conll], \
                    [[token["misc"]["name"] for token in sentence] for sentence in test_conll]
        test_sentences = test_data[0]
        logger.info(f"{len(test_sentences)} test texts")
        average_length = np.mean([len(t) for t in test_sentences])
        logger.info(f"Average test text length: {average_length:.3f} words")
        logger.info("Finished loading the test dataset")

        logger.info("Inferring embeddings for the test set...")
        x_test = infer_embeddings(test_sentences, el_model, layers=elmo_layers)

        predictions = model.predict(x_test)
        # map predictions to the binary {0, 1} range:
        predictions = np.around(predictions)

        # Convert predictions from integers back to text labels:
        predictions = [classes[int(np.argmax(pred))] for pred in predictions]

        counter = 0
        for nr, sentence in enumerate(test_conll):
            for token in sentence:
                token["misc"]["name"] = predictions[counter]
                counter += 1

        PRED_FILE = args.name
        with open(PRED_FILE, "w") as f:
            for sentence in test_conll:
                f.write(sentence.serialize())
        logger.info(f"Predictions written to {PRED_FILE}")
