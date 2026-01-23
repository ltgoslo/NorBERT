import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer

STEP_PATTERN = re.compile(r"\d+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_directory', type=str, default='/cluster/work/projects/nn9851k/mariiaf/hplt/')
    parser.add_argument('--output_model_directory', type=str, default='/cluster/work/projects/nn9851k/mariiaf/hplt/hplt_hf_models')
    parser.add_argument('--language', type=str, default='deu_Latn')
    parser.add_argument('--all_checkpoints', action='store_true')
    parser.add_argument('--model_directory', default="norbert_2_nodes/")
    args = parser.parse_args()
    return args


def convert_to_hf(
        input_model_directory,
        output_model_directory,
        language,
        all_checkpoints,
        model_directory,
):
    checkpointing_steps = [6250]
    checkpoints_directory = os.path.join(
        input_model_directory, language, model_directory, f"NorBERT4_base_{language}_2_nodes",
    )
    if all_checkpoints:
        print(f"Files in the checkpoints_directory: {os.listdir(checkpoints_directory)}")
        checkpointing_steps = [
            int(re.search(STEP_PATTERN, bin_name).group(0)) for bin_name in os.listdir(checkpoints_directory)
        ]
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

        prototype_directory = "huggingface_prototype"
        os.system(f"cp {prototype_directory}/* {step_output_model_directory}")

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in checkpoint['model'].items():
            new_state_dict[k.removeprefix("_orig_mod.")] = v
        torch.save(new_state_dict, os.path.join(step_output_model_directory, "pytorch_model.bin"))

        os.system(f"cp {input_model_directory}/{language}/tokenizer.json {step_output_model_directory}")


def main():
    args = parse_args()
    convert_to_hf(
        args.input_model_directory,
        args.output_model_directory,
        args.language,
        args.all_checkpoints,
        args.model_directory,
    )


if __name__ == "__main__":
    main()