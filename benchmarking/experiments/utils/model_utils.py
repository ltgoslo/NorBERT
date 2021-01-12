import tensorflow as tf
from transformers import (TFBertForSequenceClassification, BertTokenizer,
                         TFXLMRobertaForSequenceClassification, XLMRobertaTokenizer,
                         TFBertForTokenClassification, TFXLMRobertaForTokenClassification)

import sys
sys.path.append("..")
from data_preparation.data_preparation_pos import MBERT_Tokenizer, XLMR_Tokenizer
from utils.pos_utils import ignore_acc

models = {
    "mbert": {
        "pos": TFBertForTokenClassification.from_pretrained,
        "sentiment": TFBertForSequenceClassification.from_pretrained
    },
    "xlm-roberta": {
        "pos": TFXLMRobertaForTokenClassification.from_pretrained,
        "sentiment": TFXLMRobertaForSequenceClassification.from_pretrained
    }
}

tokenizers = {
    "mbert": {
        "pos": MBERT_Tokenizer.from_pretrained,
        "sentiment": BertTokenizer.from_pretrained
    },
    "xlm-roberta": {
        "pos": XLMR_Tokenizer.from_pretrained,
        "sentiment": XLMRobertaTokenizer.from_pretrained
    }
}

def set_tf_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

def get_full_model_names(short_model_name):
    d = {
        "bert-base-multilingual-cased": "bert-base-multilingual-cased",
        "tf-xlm-roberta-base": "jplu/tf-xlm-roberta-base",
        "mbert": "bert-base-multilingual-cased",
        "xlm-roberta": "tf-xlm-roberta-base"
    }
    if short_model_name in d.values():
        # In case the long model name (eg. 'tf-xlm-roberta-base' is given)
        # Return only the full name
        return d[short_model_name]
    else:
        model_name = d[short_model_name]
        return model_name, d[model_name]

def create_model(short_model_name, task, num_labels):
    return (models[short_model_name][task](get_full_model_names(short_model_name)[1],
                                          num_labels=num_labels),
            get_tokenizer(short_model_name, task))

def get_tokenizer(short_model_name, task):
    return tokenizers[short_model_name][task](get_full_model_names(short_model_name)[1])

def compile_model(model, task, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def make_batches(dataset, batch_size, repetitions, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(int(1e6), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    n_batches = len(list(dataset.as_numpy_iterator()))
    dataset = dataset.repeat(repetitions)
    return dataset, n_batches
