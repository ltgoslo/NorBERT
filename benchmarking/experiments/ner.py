# /bin/env python3
# coding: utf-8

# Adapted from
# https://colab.research.google.com/gist/peregilk/6f5efea432e88199f5d68a150cef237f/-nbailab-finetuning-and-evaluating-a-bert-model-for-ner-and-pos.ipynb

import os
import numpy as np
from datasets import load_dataset, load_from_disk
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
import argparse

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


# Preprocessing the dataset
# Tokenize texts and align the labels with them.


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if label_all_tokens else -100
                )
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Metrics
def compute_metrics(pairs):
    current_predictions, labels = pairs
    current_predictions = np.argmax(current_predictions, axis=2)

    # Remove ignored index (special tokens)
    real_predictions = [
        [label_list[p] for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(current_predictions, labels)
    ]
    true_labels = [
        [label_list[lab] for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(current_predictions, labels)
    ]

    return {
        "accuracy_score": accuracy_score(true_labels, real_predictions),
        "precision": precision_score(true_labels, real_predictions),
        "recall": recall_score(true_labels, real_predictions),
        "f1": f1_score(true_labels, real_predictions),
        "report": classification_report(true_labels, real_predictions, digits=4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="ltgoslo/norbert")
    parser.add_argument("--dataset", default="norne_nob")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset
    task_name = "ner"

    overwrite_cache = True

    seed = 42
    set_seed(seed)

    # Tokenizer
    padding = False
    max_length = 512
    label_all_tokens = False

    # Training
    num_train_epochs = args.epochs  # @param {type: "number"}
    per_device_train_batch_size = 8  # param {type: "integer"}
    per_device_eval_batch_size = 8  # param {type: "integer"}
    learning_rate = 3e-05  # @param {type: "number"}
    weight_decay = 0.0  # param {type: "number"}
    adam_beta1 = 0.9  # param {type: "number"}
    adam_beta2 = 0.999  # param {type: "number"}
    adam_epsilon = 1e-08  # param {type: "number"}
    max_grad_norm = 1.0  # param {type: "number"}
    num_warmup_steps = 750  # @param {type: "number"}
    save_total_limit = 1  # param {type: "integer"}
    load_best_model_at_end = True  # @param {type: "boolean"}

    output_dir = model_name.split("/")[-1] + "_" + dataset_name + "_" + str(per_device_train_batch_size)
    overwrite_output_dir = False

    # Load the dataset
    dataset = load_from_disk(dataset_name)
    print(f"Dataset loaded from {dataset_name}")

    # Getting some variables from the dataset
    column_names = dataset["train"].column_names
    features = dataset["train"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{task_name}_tags" if f"{task_name}_tags" in column_names else column_names[1]
    )
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    num_labels = len(label_list)

    # Look at the dataset
    print(f"###Quick Look at the NorNE Dataset")
    print(dataset["train"].data.to_pandas()[[text_column_name, label_column_name]])

    print(f"###All labels ({num_labels})")
    print(label_list)

    if task_name == "ner":
        mlabel_list = {label.split("-")[-1] for label in label_list}
        print(f"###Main labels ({len(mlabel_list)})")
        print(mlabel_list)

    """# Initialize Training"""

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=task_name,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        load_from_cache_file=not overwrite_cache,
        num_proc=os.cpu_count(),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        warmup_steps=num_warmup_steps,
        load_best_model_at_end=load_best_model_at_end,
        seed=seed,
        save_total_limit=save_total_limit,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    """# Start Training"""

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )
    #
    # Print Results
    output_train_file = os.path.join(output_dir, "train_results.txt")
    with open(output_train_file, "w") as writer:
        print("**Train results**")
        for key, value in sorted(train_result.metrics.items()):
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    """# Evaluate the Model"""

    print("**Evaluate**")
    results = trainer.evaluate()

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        print("**Eval results**")
        for key, value in results.items():
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    """# Run Predictions on the Test Dataset"""

    print("**Predict**")
    test_dataset = tokenized_datasets["test"]
    predictions, our_labels, metrics = trainer.predict(test_dataset)
    predictions = np.argmax(predictions, axis=2)

    output_test_results_file = os.path.join(output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        print("**Predict results**")
        for key, value in sorted(metrics.items()):
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, our_labels)
    ]

    # Save predictions
    output_test_predictions_file = os.path.join(output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        for prediction in true_predictions:
            writer.write(" ".join(prediction) + "\n")
