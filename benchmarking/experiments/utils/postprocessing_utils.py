#!/bin/env python3

import numpy as np
import pandas as pd
import utils.utils as utils


def find_training_langs(table):
    return [col_name for col_name in table.columns if
            (table[col_name].apply(lambda x: isinstance(x, (np.floating, float))).all())]


def reorder_columns(table):
    lang_column = utils.find_lang_column(table)
    training_langs = find_training_langs(table)
    training_langs.sort()
    testing_langs = table[lang_column].values.tolist()
    testing_langs.sort()
    assert training_langs == testing_langs, "Training language columns are missing"
    return table[[lang_column] + table[lang_column].values.tolist()]


def fill_missing_columns(table):
    training_langs = find_training_langs(table)
    missing_langs = np.setdiff1d(table[utils.find_lang_column(table)], training_langs)
    table[missing_langs] = pd.DataFrame([[np.nan] * len(missing_langs)], index=table.index)
    return table


def mean_exclude_by_group(table):
    table_by_test_group = pd.DataFrame(
        {"Group": ["Fusional", "Isolating", "Agglutinative", "Introflexive"]})

    for train_lang in find_training_langs(table):
        metric_avgs = []
        for lang_group in table_by_test_group["Group"]:
            avg = table[(table["Group"] == lang_group) & (table["Language"] != train_lang)][
                train_lang].mean()
            metric_avgs.append(avg)
        table_by_test_group[train_lang] = metric_avgs

    return table_by_test_group


def mean_exclude(table):
    lang_cols = table.columns[1:]
    means = []
    for i, row in table.iterrows():
        row_mean = row[[col for col in lang_cols if col != row.iloc[0]]].mean()
        means.append(row_mean)
    return means


def retrieve_results(file_path, skip):
    results = pd.read_excel(file_path, sheet_name=None, header=None)
    output = {}

    for metric, df in results.items():
        table_names = [
            "langvlang",
            "langvgroup",
            "groupvgroup",
        ]

        tables = {}
        start = 0
        end = df.shape[1] - 1
        for name in table_names:
            temp = df.loc[start:end]
            temp.columns = temp.iloc[0].values
            temp = temp.drop(temp.index[0])
            start = end + skip + 1
            end = start + 6
            tables[name] = temp.reset_index(drop=True)
        output[metric] = tables
    return output
