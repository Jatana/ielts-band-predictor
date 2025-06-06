from __future__ import annotations

from pathlib import Path

import dvc.api
import pandas as pd
from datasets import Dataset, DatasetDict
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ielts_band_predictor.scripts.remove_nonascii import strip_non_ascii


class IELTSDataModule(LightningDataModule):
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
    ):
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

    def prepare_data(self):
        if self.raw_path.startswith("s3://") or ".dvc" in self.raw_path:
            with dvc.api.open(self.raw_path, mode="r") as f:
                _ = f.readline()
        else:
            pass

    def setup(self, stage):
        if self.dataset is not None:
            return

        url = self.raw_path
        if ".dvc" in self.raw_path:
            url = dvc.api.get_url(self.raw_path)

        path: str | Path = url
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
                continue  # discard this row entirely
            clean_texts.append(clean)
            labels.append(float(band))

        df = pd.DataFrame({"text": clean_texts, "labels": labels})

        # df["text"] = df.apply(
        #     lambda row: f'PROMPT: {row["prompt"]}  ESSAY: {row["essay"]}',
        #     axis=1,
        # )

        # stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=self.val_size,
            stratify=df["labels"],
            random_state=self.seed,
        )
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
        self.dataset = DatasetDict(train=train_ds, validation=val_ds)

        # tokenizer
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

        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # dataloaders
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
