import os
from typing import Dict, Union

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    base_url: str = Field(
        default='',
    )
    model: str = Field(
        default='',
    )
    # так делать плохо
    password: str = Field(
        default=''
    )
    rag_settings: dict = Field(
        default=dict()
    )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        data = cls()._load_yaml(path)
        return cls(**data)

    def _load_yaml(self, path: str) -> Dict[str, Union[str, bool, int, float]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        try:
            with open(path, 'r', encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"Failed to load YAML from {path}")
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {e}")
