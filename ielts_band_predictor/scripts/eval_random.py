from __future__ import annotations

import json
import math
import random
import statistics
import textwrap
from io import StringIO
from pathlib import Path
from typing import List

import dvc.api
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from ielts_band_predictor.models import BertBandRegressor
from ielts_band_predictor.scripts.remove_nonascii import strip_non_ascii


def _load_jsonl(path: str | Path):
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f]


def _sample(records: List[dict], k: int):
    return random.sample(records, k=min(k, len(records)))


def _wrap(text: str, width: int = 88):
    return textwrap.fill(text, width=width, replace_whitespace=False)


@hydra.main(config_path="../../configs", config_name="eval_random", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # load model & tokenizer
    model = BertBandRegressor.load_from_checkpoint(cfg.ckpt, map_location="cpu").eval()
    tokenizer = AutoTokenizer.from_pretrained(model.hparams.pretrained_name)
    print("Loaded model:", model.hparams.pretrained_name)

    # sample essays
    content = dvc.api.read(cfg.data, mode="r")
    df = pd.read_json(StringIO(content), lines=True)
    dataset = df.to_dict(orient="records")
    batch = _sample(dataset, cfg.k)

    abs_errs, sq_errs = [], []

    for i, rec in enumerate(batch, 1):
        rec["prompt"] = strip_non_ascii(
            rec["prompt"],
            ratio=cfg.cleaning.max_non_ascii_ratio,
            absolute=cfg.cleaning.max_non_ascii_abs,
        )
        rec["essay"] = strip_non_ascii(
            rec["essay"],
            ratio=cfg.cleaning.max_non_ascii_ratio,
            absolute=cfg.cleaning.max_non_ascii_abs,
        )
        if not rec["prompt"] or not rec["essay"]:
            print("Skipped: too many non-ASCII symbols\n")
            continue

        text = f'PROMPT: {rec["prompt"]}  ESSAY: {rec["essay"]}'
        enc = tokenizer(
            text, truncation=True, padding="max_length", max_length=cfg.max_len, return_tensors="pt"
        )
        with torch.no_grad():
            pred = model(**enc).item()

        true = float(rec["band"])
        err = abs(pred - true)
        abs_errs.append(err)
        sq_errs.append(err**2)

        print(f"\n#{i:02d}  True={true:.1f}  Pred={pred:.2f}  |err|={err:.2f}")
        print("Prompt:", _wrap(rec["prompt"]))
        print("Essay :\n", _wrap(rec["essay"][:600]), "..." if len(rec["essay"]) > 600 else "")

    mae, rmse = statistics.mean(abs_errs), math.sqrt(statistics.mean(sq_errs))
    print("\n" + "-" * 60)
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")


if __name__ == "__main__":
    main()
