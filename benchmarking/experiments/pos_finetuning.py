#!/bin/env python3

import argparse
import numpy as np
import pandas as pd
import data_preparation.data_preparation_pos as data_preparation_pos
import fine_tuning
import utils.model_utils as model_utils
import utils.pos_utils as pos_utils


def test(training_lang,
         test_lang,
         split="test",
         short_model_name="ltgoslo/norbert",
         model_name="norbert", epochs=5, data_path=None, task="pos"):
    checkpoints_path = "checkpoints/"
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
    # Model parameters
    max_length = 256
    batch_size = 8
    eval_batch_size = 8
    learning_rate = 2e-5
    tagset = pos_utils.get_ud_tags()
    num_labels = len(tagset)
    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels, tagset=tagset,
                        eval_batch_size=64)
    weights_path = checkpoints_path + training_lang + "/"
    weights_filename = model_name + "_pos.hdf5"
    print("Using weights from", weights_path + weights_filename)
    trainer.model.load_weights(weights_path + weights_filename)
    # Checkpoint for best model weights
    # trainer.setup_checkpoint(checkpoints_path)
    trainer.prepare_data()
    #
    test_lang_path = data_path + test_lang
    test_data, test_dataset = data_preparation_pos.load_dataset(test_lang_path, trainer.tokenizer,
                                                                max_length, trainer.tagset,
                                                                dataset_name=split)
    trainer.setup_eval(test_data, split)
    test_dataset, test_batches = model_utils.make_batches(test_dataset, eval_batch_size,
                                                          repetitions=1, shuffle=False)
    # return trainer, test_data, test_dataset, test_batches
    test_preds = trainer.handle_oom(trainer.model.predict,
                                    test_dataset,
                                    steps=test_batches,
                                    verbose=1)
    score = trainer.metric(test_preds, test_data, split) * 100
    print("{0}-{1} {2}: {3:.1f}".format(training_lang, test_lang, split, score))
    return score


def train(training_lang,
          short_model_name="ltgoslo/norbert",
          epochs=10, data_path=None, task="pos"):
    checkpoints_path = "checkpoints/"
    #
    trainer = fine_tuning.Trainer(training_lang, data_path, task, short_model_name)
    #
    # Model parameters
    max_length = 256
    batch_size = 8
    learning_rate = 2e-5
    tagset = pos_utils.get_ud_tags()
    num_labels = len(tagset)
    #
    # Model creation
    trainer.build_model(max_length, batch_size, learning_rate, epochs, num_labels, tagset=tagset,
                        eval_batch_size=64)
    #
    # Checkpoint for best model weights
    trainer.setup_checkpoint(checkpoints_path)
    #
    trainer.prepare_data()
    #
    print("Train examples:", len(trainer.train_data))
    #
    # Print an example sentence for sanity
    example_batch = trainer.train_dataset.as_numpy_iterator().next()
    for token, label in zip(example_batch[0]["input_ids"][0], example_batch[1][0]):
        if not token:
            break
        elif token == example_batch[0]["input_ids"][0][10]:
            print("...")
            break
        print("{:<25}{:<20}".format(trainer.tokenizer.decode(int(token)), tagset[label]))
    #
    try:
        trainer.setup_training()
        trainer.train()
        trainer.make_definitive()
    except KeyboardInterrupt:
        pass
    return trainer


def prepare_test_data(trainer):
    # Load plain data and TF dataset
    data, dataset = data_preparation_pos.load_dataset(
        trainer.lang_path, trainer.tokenizer, trainer.max_length, trainer.tagset,
        dataset_name="test")
    trainer.setup_eval(data, "test")
    dataset, batches = model_utils.make_batches(
        dataset, trainer.eval_batch_size, repetitions=1, shuffle=False)
    return dataset, batches, data


def setup_eval(data, tokenizer, label_map, max_length, dataset_name="test"):
    eval_info = {dataset_name: {}}
    eval_info[dataset_name]["all_words"] = []
    eval_info[dataset_name]["all_labels"] = []
    eval_info[dataset_name]["real_tokens"] = []
    eval_info[dataset_name]["subword_locs"] = []
    acc_lengths = 0
    #
    for i in range(len(data)):
        eval_info[dataset_name]["all_words"].extend(data[i]["tokens"])  # Full words
        eval_info[dataset_name]["all_labels"].extend(
            [label_map[label] for label in data[i]["tags"]])
        _, _, idx_map = tokenizer.subword_tokenize(data[i]["tokens"], data[i]["tags"])
        # Examples always start at a multiple of max_length
        # Where they end depends on the number of resulting subwords
        example_start = i * max_length
        example_end = example_start + len(idx_map)
        eval_info[dataset_name]["real_tokens"].extend(
            np.arange(example_start, example_end, dtype=int))
        # Get subword starts and ends
        sub_ids, sub_starts, sub_lengths = np.unique(idx_map, return_counts=True, return_index=True)
        sub_starts = sub_starts[sub_lengths > 1] + acc_lengths
        sub_ends = sub_starts + sub_lengths[sub_lengths > 1]
        eval_info[dataset_name]["subword_locs"].extend(np.array([sub_starts, sub_ends]).T.tolist())
        acc_lengths += len(idx_map)
    return eval_info


def get_score_pos(preds, dataset_name, eval_info):
    filtered_preds = preds[0].argmax(axis=-1).flatten()[
        eval_info[dataset_name]["real_tokens"]].tolist()
    filtered_logits = preds[0].reshape(
        (preds[0].shape[0] * preds[0].shape[1], preds[0].shape[2])
    )[eval_info[dataset_name]["real_tokens"]]
    new_preds = pos_utils.reconstruct_subwords(
        eval_info[dataset_name]["subword_locs"], filtered_preds, filtered_logits
    )
    return (np.array(eval_info[dataset_name]["all_labels"]) == np.array(new_preds)).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="norbert")
    parser.add_argument("--short_model_name", default="ltgoslo/norbert")

    args = parser.parse_args()

    training_language = "no"
    ud_data_path = "../data/ud/"
    run_name = args.model_name
    model_identifier = args.short_model_name
    current_task = "pos"

    # # Model parameters
    # tagset = ["O", "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    #           "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    # num_labels = len(tagset)
    # label_map = {label: i for i, label in enumerate(tagset)}

    # Train models
    training_object = train(training_language, short_model_name=model_identifier,
                            data_path=ud_data_path, task=current_task)

    dev_score = test(training_language,
                     training_language,
                     "dev",
                     short_model_name=model_identifier,
                     model_name=run_name, data_path=ud_data_path, task=current_task)

    test_score = test(training_language,
                      training_language,
                      "test",
                      short_model_name=model_identifier,
                      model_name=run_name,
                      data_path=ud_data_path, task=current_task)

    table = pd.DataFrame({"Language": training_language,
                          "Dev Accuracy": [dev_score],
                          "Test Accuracy": [test_score]
                          })

    print(table)
    print(table.to_latex(index=False, float_format="{0:.1f}".format))
    table.to_csv("results/{}_pos.tsv".format(run_name), sep="\t")
    print(f"Scores saved to results/{run_name}_pos.tsv")
