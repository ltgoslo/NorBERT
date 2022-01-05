# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Modified by David Samuel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert BERT checkpoints trained with TensorFlow 2 to PyTorch."""

import os
import argparse
import torch
import tensorflow as tf
from transformers import BertConfig, BertForPreTraining


def assign(layer, tensor):
    assert layer.data.shape == tensor.shape
    layer.data = tensor


def load_tf2_weights_in_bert(model, config, tf_checkpoint_path):
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)

    names, arrays = [], []
    for full_name, shape in init_vars:
        name = full_name.split("/")
        if full_name == "_CHECKPOINTABLE_OBJECT_GRAPH" or name[0] in [
            "global_step",
            "save_counter",
        ]:
            # print(f'Skipping non-model layer {full_name}')
            continue
        if "optimizer" in full_name:
            # print(f'Skipping optimization layer {full_name}')
            continue

        assert name[0] == "model"
        assert name[1] == "layer_with_weights-0"
        assert name[2].startswith("layer_with_weights")
        assert name[3].startswith("layer_with_weights")

        name = name[2:]

        print("Loading TF weight {} with shape {}".format(full_name, shape))

        # read data
        array = tf.train.load_variable(tf_path, full_name)
        names.append(name)
        arrays.append(torch.from_numpy(array))

    print(f"Read a total of {len(arrays)} layers")

    # convert layers
    print("Converting weights...")
    for name, array in zip(names, arrays):
        module = {0: "encoder", 1: "mlm", 2: "nsp"}[int(name[0].split("-")[-1])]
        n_layer = int(name[1].split("-")[-1])

        if module == "encoder":
            # embedding modules
            if n_layer == 0:
                assert name[2] == "embeddings"
                assign(model.bert.embeddings.word_embeddings.weight, array)
                assign(
                    model.cls.predictions.decoder.weight, array
                )  # this weight matrix is shared
            elif n_layer == 1:
                assert name[2] == "embeddings"
                assign(model.bert.embeddings.position_embeddings.weight, array)
            elif n_layer == 2:
                assert name[2] == "embeddings"
                assign(model.bert.embeddings.token_type_embeddings.weight, array)
            elif n_layer == 3:
                if name[2] == "beta":
                    assign(model.bert.embeddings.LayerNorm.bias, array)
                elif name[2] == "gamma":
                    assign(model.bert.embeddings.LayerNorm.weight, array)
                else:
                    raise Exception(
                        f"Layer 3 should be either beta or gamma, not {name}"
                    )

            # transformer layers
            elif n_layer < config.num_hidden_layers + 4:
                submodule = model.bert.encoder.layer[n_layer - 4]

                if name[2] == "_attention_layer":
                    submodule = submodule.attention.self
                    if name[3] == "_key_dense":
                        submodule = submodule.key
                    elif name[3] == "_query_dense":
                        submodule = submodule.query
                    elif name[3] == "_value_dense":
                        submodule = submodule.value
                    else:
                        raise Exception(
                            f"{n_layer - 4}th attention layer should be either a key, query or value, not {name}"
                        )

                    if name[4] == "bias":
                        assign(submodule.bias, array.flatten())
                    elif name[4] == "kernel":
                        assign(
                            submodule.weight, array.flatten(start_dim=1, end_dim=2).T
                        )
                    else:
                        raise Exception(
                            f"{n_layer - 4}th attention's {name[3]} should be either a bias or a kernel, not {name}"
                        )

                elif name[2] == "_attention_layer_norm":
                    if name[3] == "beta":
                        assign(submodule.attention.output.LayerNorm.bias, array)
                    elif name[3] == "gamma":
                        assign(submodule.attention.output.LayerNorm.weight, array)
                    else:
                        raise Exception(
                            f"{n_layer - 4}th attention's LayerNorm should be either beta or gamma, not {name}"
                        )

                elif name[2] == "_attention_output_dense":
                    if name[3] == "bias":
                        assign(submodule.attention.output.dense.bias, array.flatten())
                    elif name[3] == "kernel":
                        assign(
                            submodule.attention.output.dense.weight,
                            array.flatten(start_dim=0, end_dim=1).T,
                        )
                    else:
                        raise Exception(
                            f"{n_layer - 4}th attention's {name[2]} should be either a bias or a kernel, not {name}"
                        )

                elif name[2] == "_intermediate_dense":
                    if name[3] == "bias":
                        assign(submodule.intermediate.dense.bias, array)
                    elif name[3] == "kernel":
                        assign(submodule.intermediate.dense.weight, array.T)
                    else:
                        raise Exception(
                            f"{n_layer - 4}th intermediate should be either a bias or a kernel, not {name}"
                        )

                elif name[2] == "_output_dense":
                    if name[3] == "bias":
                        assign(submodule.output.dense.bias, array)
                    elif name[3] == "kernel":
                        assign(submodule.output.dense.weight, array.T)
                    else:
                        raise Exception(
                            f"{n_layer - 4}th output should be either a bias or a kernel, not {name}"
                        )

                elif name[2] == "_output_layer_norm":
                    if name[3] == "beta":
                        assign(submodule.output.LayerNorm.bias, array)
                    elif name[3] == "gamma":
                        assign(submodule.output.LayerNorm.weight, array)
                    else:
                        raise Exception(
                            f"{n_layer - 4}th output's LayerNorm should be either beta or gamma, not {name}"
                        )

                else:
                    raise Exception(
                        f"Layer {n_layer} should be a Transformer layer, not {name}"
                    )

            elif n_layer == config.num_hidden_layers + 4:
                if name[2] == "bias":
                    assign(model.bert.pooler.dense.bias, array)
                elif name[2] == "kernel":
                    assign(model.bert.pooler.dense.weight, array.T)
                else:
                    raise Exception(
                        f"Pooler should be either a bias or a kernel, not {name}"
                    )

        elif module == "mlm":
            if n_layer == 0:
                if name[2] == "bias":
                    assign(model.cls.predictions.transform.dense.bias, array)
                elif name[2] == "kernel":
                    assign(model.cls.predictions.transform.dense.weight, array.T)
                else:
                    raise Exception(
                        f"MLM transformation should be either a bias or a kernel, not {name}"
                    )
            elif n_layer == 1:
                if name[2] == "beta":
                    assign(model.cls.predictions.transform.LayerNorm.bias, array)
                elif name[2] == "gamma":
                    assign(model.cls.predictions.transform.LayerNorm.weight, array)
                else:
                    raise Exception(
                        f"MLM transformation's LayerNorm should be either beta or gamma, not {name}"
                    )
            elif n_layer == 2:
                if name[2] == "bias":
                    assign(model.cls.predictions.decoder.bias, array)
                else:
                    raise Exception(
                        f"MLM classification should be either a bias, not {name}"
                    )
            else:
                raise Exception(f"MLM classification has only three layers, not {name}")

        elif module == "nsp":
            assert n_layer == 0
            if name[2] == "bias":
                assign(model.cls.seq_relationship.bias, array)
            elif name[2] == "kernel":
                assign(model.cls.seq_relationship.weight, array.T)
            else:
                raise Exception(
                    f"NSP classification should be either a bias or a kernel, not {name}"
                )

        else:
            raise Exception(f"Non-existent module: {module}")


def convert_tf_checkpoint_to_pytorch(
    tf_checkpoint_path, bert_config_file, pytorch_dump_path
):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf2_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path
    )
