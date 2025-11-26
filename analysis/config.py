from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, ValidationError, model_validator
import tomli as tomllib


class Paths(BaseModel):
    data: Path = Field(...)
    output: Path = Field(...)
    cache: Optional[Path] = Field(None)

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Paths":
        self.output.mkdir(parents=True, exist_ok=True)
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        return self


class Metadata(BaseModel):
    date_column: str = Field(...)
    entity_columns: List[str] = Field(...)
    policy_columns: List[str]
    outcome_columns: List[str]
    extra_columns: List[str] = Field(default_factory=list)

    @property
    def id_columns(self) -> List[str]:
        return [*self.entity_columns, self.date_column]


class HParamSearch(BaseModel):
    seed: int = 1
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0


class AnalysisSettings(BaseModel):
    paths: Paths
    metadata: Metadata
    hparams: HParamSearch

    @classmethod
    def _apply_env(cls, cfg: Dict[str, Any], prefix: str = "ANALYSIS_") -> None:
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            parts = env_key[len(prefix):].split("__")
            node: Dict[str, Any] = cfg
            for part in parts[:-1]:
                node = node.setdefault(part.lower(), {})
            node[parts[-1].lower()] = env_val

    @classmethod
    def load(cls, file_path: Union[str, os.PathLike, None] = None) -> "AnalysisSettings":
        cfg_path = Path(file_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("rb") as f:
            data: Dict[str, Any] = tomllib.load(f)

        cls._apply_env(data)

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration:\n{e}") from e


class _classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


class AnalysisConfig:
    _settings: Optional[AnalysisSettings] = None

    @classmethod
    def load(cls, file_path: Union[str, os.PathLike, None] = "config.toml") -> None:
        print("Loading config...")
        cls._settings = AnalysisSettings.load(file_path=file_path)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._settings is None:
            cls.load()

    @_classproperty
    def paths(cls) -> Paths:
        cls._ensure_loaded()
        return cls._settings.paths 

    @_classproperty
    def metadata(cls) -> Metadata:
        cls._ensure_loaded()
        return cls._settings.metadata

    @_classproperty
    def hparams(cls) -> HParamSearch:
        cls._ensure_loaded()
        return cls._settings.hparams
