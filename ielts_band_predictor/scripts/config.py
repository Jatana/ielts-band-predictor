# ielts_band_predictor/config.py
# ------------------------------------------------------------
# Dataclass schemas for Hydra configs   (Hydra 1.3 / Ω-Conf 2.x)
# ------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Union


# ---------------------- DataModule --------------------------
@dataclass
class DataModuleCfg:
    _target_: str = "ielts_band_predictor.data.IELTSDataModule"

    # paths / DVC
    raw_path: str = "data/raw/essays.jsonl"

    # hyper-parameters
    batch_size: int = 16
    max_length: int = 512
    val_size: float = 0.10
    num_workers: int = 4
    seed: int = 42

    # will be overridden from model.pretrained_name in YAML
    tokenizer_name: str = "bert-base-uncased"


# ------------------------ Model -----------------------------
@dataclass
class ModelCfg:
    _target_: str = "ielts_band_predictor.models.BertBandRegressor"

    pretrained_name: str = "bert-base-uncased"
    lr: float = 2e-5
    weight_decay: float = 1e-2
    freeze_n_layers: int = 8  # how many bottom BERT layers to freeze


# ----------------------- Trainer ----------------------------
@dataclass
class TrainerCfg:
    accelerator: str = "gpu"  # "cpu", "gpu", "mps"
    devices: Union[int, List[int]] = 1  # 1 GPU or a list like [0,1]
    max_epochs: int = 5
    precision: int = 16  # AMP fp16
    log_every_n_steps: int = 20
    deterministic: bool = True


# --------------------- Root schema --------------------------
@dataclass
class Config:
    """Top-level config Hydra passes into train.py"""

    seed: int = 42

    # Nested groups (populated from YAML defaults:)
    datamodule: DataModuleCfg = field(default_factory=DataModuleCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)

    # These two are *overwritten* by YAML variants (optimizer/ callbacks lists)
    optimizer: Any = None
    callbacks: Any = field(default_factory=list)

    # Optional extras (add later if needed)
    scheduler: Any = None
    logger: Any = None
