from __future__ import annotations

import json
from pathlib import Path
import copy
from transformers.configuration_utils import PretrainedConfig


class GptBertConfig(PretrainedConfig):

    def __init__(
        self,
        config_file: Path | str | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = "norbert4"

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
