#! python3
# coding: utf-8

import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--path", required=True, help="Path to the logs directory", action="store"
    )
    args = parser.parse_args()

    logfiles = [f for f in os.scandir(args.path) if f.name.endswith("log.tsv")]
    language = args.path.split("/")[-1]

    task = None
    logs = {}
    for f in logfiles:
        modelname = "_".join(f.name.split("_")[:-2])
        task = f.name.split("_")[-2]
        logs[modelname] = pd.read_csv(f, sep="\t", header=0)

    color = iter(cm.rainbow(np.linspace(0, 1, len(logs))))

    for model in sorted(logs):
        epochs = logs[model].epoch
        train_scores = logs[model].train_score
        dev_scores = logs[model].dev_score
        c = next(color)
        plt.plot(epochs, train_scores, label=model+"_train", linestyle='dashed', c=c)
        plt.plot(epochs, dev_scores, label=model+"_dev", marker='o', c=c)

    plt.xlabel("Epochs")
    plt.ylabel("Performance")
    plt.xticks(range(20))
    plt.legend(loc="best")
    plt.title(f"Sentiment classification")
    plt.savefig(f"{task}_{language}_plot.png", dpi=300, bbox_inches="tight")
