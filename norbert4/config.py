from __future__ import annotations

import json
from pathlib import Path
import copy


class ModelConfig:

    def __init__(
        self,
        config_file: Path | str | None = None,
        **kwargs
    ):
        self.model: str

        # General information
        self.model = "base"

        # Vocabulary
        self.vocab_size = 16384
        self.max_sequence_length = 512

        # Model dimensions
        self.hidden_size = 768
        self.intermediate_size = 2048
        self.num_attention_heads = 12
        self.num_layers = 12
        self.d_qk = 64

        # Dropout probabilities
        self.embedding_dropout_p = 0.1
        self.attention_probabilities_dropout_p = 0.1
        self.attention_output_dropout_p = 0.1
        self.feed_forward_dropout_p = 0.1

        # Position Emebedding
        self.rope_theta = 160_000

        # Norms
        self.word_norm_eps = 1e-7
        self.word_norm_affine = False

        self.attention_pre_norm_eps = 1e-7
        self.attention_pre_norm_affine = False

        self.attention_inter_norm_eps = 1e-7
        self.attention_inter_norm_affine = True

        self.feed_forward_pre_norm_eps = 1e-7
        self.feed_forward_pre_norm_affine = False

        self.feed_forward_inter_norm_eps = 1e-7
        self.feed_forward_inter_norm_affine = False

        self.classifier_pre_norm_eps = 1e-7
        self.classifier_pre_norm_affine = False

        self.classifier_post_norm_eps = 1e-7
        self.classifier_post_norm_affine = False

        if config_file is not None:
            if type(config_file) is str:
                config_file = Path(config_file)
            assert type(config_file) is not Path, "The config_file should either be a Path or str"
            with config_file.open("r") as file:
                config = json.load(file)

            for attr, value in config.items():
                if isinstance(value, str):
                    value = value.lower()
                setattr(self, attr, value)

        for attr, value in kwargs.items():
            if isinstance(value, str):
                value = value.lower()
            setattr(self, attr, value)

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output: dict

        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Path | str) -> None:
        """Save this instance to a json file."""
        if isinstance(json_file_path, str):
            json_file_path: Path = Path(json_file_path)
        with json_file_path.open("w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

    @classmethod
    def create_base_config(cls, json_file_path: Path | str | None = None) -> ModelConfig:
        config: ModelConfig

        config = ModelConfig()
        if json_file_path is not None:
            config.to_json_file(json_file_path)

        return config
