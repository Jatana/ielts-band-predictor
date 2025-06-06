# ielts_band_predictor/data.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import dvc.api
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ielts_band_predictor.scripts.remove_nonascii import strip_non_ascii


class IELTSDataModule(LightningDataModule):
    """
    Lightning-обёртка над HF-Dataset, настраиваемая через Hydra-config.

    Параметры поступают из configs/datamodule/*.yaml, т. е. вам ничего не
    нужно менять в коде — только в конфиге.
    """

    def __init__(
        self,
        raw_path: str,
        batch_size: int = 16,
        max_length: int = 512,
        val_size: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
        max_non_ascii_ratio: float = 0.05,
        max_non_ascii_abs: int = 50,
        tokenizer_name: str = "bert-base-uncased",
    ) -> None:
        super().__init__()
        self.raw_path = raw_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.val_size = val_size
        self.num_workers = num_workers
        self.seed = seed
        self.max_non_ascii_ratio = 0.05
        self.max_non_ascii_abs = 50
        self.tokenizer_name = tokenizer_name

        self.tokenizer: AutoTokenizer | None = None
        self.dataset: DatasetDict | None = None

    # -------- обязательные «хуки» Lightning --------
    def prepare_data(self) -> None:
        """
        Скачивание датасета (если лежит в DVC-remote) выполняем
        только на 1 процессе: Lightning гарантирует, что метод
        вызовется лишь на rank=0.
        """
        if self.raw_path.startswith("s3://") or ".dvc" in self.raw_path:
            # пример через dvc.api.open — подтянет файл, если его нет
            with dvc.api.open(self.raw_path, mode="r") as f:
                _ = f.readline()  # только триггер для загрузки
        else:
            # локальный файл — ничего делать не надо
            pass

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None) -> None:
        """
        Загрузка и предобработка данных. Запускается и на train, и на
        inference, но только один раз в каждом процессе.
        """
        if self.dataset is not None:  # уже инициализировано
            return

        # ---------- 1. читаем jsonl ----------
        path: str | Path = dvc.api.get_url(self.raw_path) if ".dvc" in self.raw_path else self.raw_path
        df = pd.read_json(path, lines=True)

        clean_texts, labels = [], []
        for essay, prompt, band in zip(df["essay"], df["prompt"], df["band"]):
            full = f"PROMPT: {prompt}  ESSAY: {essay}"
            clean = strip_non_ascii(
                full,
                ratio=self.max_non_ascii_ratio,
                absolute=self.max_non_ascii_abs,
            )
            if clean is None:
                continue            # discard this row entirely
            clean_texts.append(clean)
            labels.append(float(band))

        df = pd.DataFrame({"text": clean_texts, "labels": labels})


        # df["text"] = df.apply(
        #     lambda row: f'PROMPT: {row["prompt"]}  ESSAY: {row["essay"]}',
        #     axis=1,
        # )

        # df = df[["text", "band"]].rename(columns={"band": "labels"})  # HF-Datasets expects 'labels'

        # ---------- 2. stratified split ----------
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_size,
            stratify=df["labels"],
            random_state=self.seed,
        )
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
        self.dataset = DatasetDict(train=train_ds, validation=val_ds)

        # ---------- 3. токенизатор ----------
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        def _tokenize(batch):
            enc = self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            enc["labels"] = batch["labels"]
            return enc

        self.dataset = self.dataset.map(_tokenize, batched=True, remove_columns=["text"])

        # задаём формат, чтобы __getitem__ выдавал torch.Tensor
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------- dataloaders --------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # (Если потребуется тест-датасет, добавьте аналогичный метод.)
