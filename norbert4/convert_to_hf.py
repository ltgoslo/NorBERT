import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer

STEP_PATTERN = re.compile(r"\d+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_directory', type=str, default='/cluster/work/projects/nn9851k/mariiaf/hplt/ltg_Latn/norbert_1_node/NorBERT4_small_ltg_Latn_1_nodes/')
    parser.add_argument('--output_model_directory', type=str, default='/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models')
    parser.add_argument('--language', type=str, default='deu_Latn')
    parser.add_argument('--all_checkpoints', action='store_true')
    parser.add_argument('--tokenizer_directory', default="/cluster/work/projects/nn9851k/mariiaf/hplt/ltg_Latn/")
    parser.add_argument('--prototype_directory', default="huggingface_prototype")
    parser.add_argument('--final_checkpoint', type=int, default=31250)
    args = parser.parse_args()
    return args


def convert_to_hf(
        input_model_directory,
        output_model_directory,
        language,
        all_checkpoints,
        tokenizer_directory,
        prototype_directory,
        final_checkpoint,
):
    checkpointing_steps = [final_checkpoint]
    checkpoints_directory = input_model_directory
    if all_checkpoints:
        print(f"Files in the checkpoints_directory: {os.listdir(checkpoints_directory)}")
        for bin_name in os.listdir(checkpoints_directory):
            step_num = re.search(STEP_PATTERN, bin_name)
            if step_num is not None:
                checkpointing_steps.append(int(step_num.group(0)))
             
        print(f"Saving steps {checkpointing_steps}")
    for step in checkpointing_steps:
        step_output_model_directory = os.path.join(output_model_directory, language+f'_{step}')
        checkpoint_path = os.path.join(
            checkpoints_directory, f"checkpoint_{step}/state_dict.bin",
        )
        if not os.path.exists(checkpoints_directory):
            raise ValueError(f"Model directory {checkpoints_directory} does not exist")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Model file {checkpoint_path} does not exist")

        if not os.path.exists(step_output_model_directory):
            os.makedirs(step_output_model_directory)

        os.system(f"cp {prototype_directory}/* {step_output_model_directory}")

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_state_dict[k.removeprefix("_orig_mod.")] = v
        torch.save(new_state_dict, os.path.join(step_output_model_directory, "pytorch_model.bin"))

        os.system(f"cp {tokenizer_directory}/tokenizer.json {step_output_model_directory}")


def main():
    args = parse_args()
    convert_to_hf(
        os.path.expanduser(args.input_model_directory),
        os.path.expanduser(args.output_model_directory),
        args.language,
        args.all_checkpoints,
        args.tokenizer_directory,
        args.prototype_directory,
        args.final_checkpoint,
    )


if __name__ == "__main__":
    main()
