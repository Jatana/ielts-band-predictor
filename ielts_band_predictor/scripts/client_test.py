from __future__ import annotations

import json
import random
import textwrap
from io import StringIO
from pathlib import Path
from typing import List

import dvc.api
import hydra
import pandas as pd
import requests
from omegaconf import DictConfig, OmegaConf

from ielts_band_predictor.scripts.remove_nonascii import strip_non_ascii


def _payload(essays: List[str]) -> bytes:
    """Build Triton-v2 REST JSON body (string tensor, shape [batch,1])."""
    return json.dumps(
        {
            "inputs": [
                {
                    "name": "TEXT",
                    "datatype": "BYTES",
                    "shape": [len(essays), 1],
                    "data": essays,
                    "parameters": {"binary_data": False},
                }
            ],
            "outputs": [{"name": "score", "parameters": {"binary_data": False}}],
        }
    ).encode("utf-8")


def query_triton(url: str, essays: List[str]) -> List[float]:
    resp = requests.post(url, data=_payload(essays), headers={"Content-Type": "application/json"})
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError("Triton request failed:\n" + resp.text) from e
    return resp.json()["outputs"][0]["data"]


def load_dataset(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open()]


def pretty_print(rec: dict, pred: float) -> None:
    print("=" * 80)
    print(f"BAND  : {rec['band']}   PREDICTED : {pred:.2f}")
    print(f"PROMPT: {rec['prompt']}")
    print("\nESSAY :\n")
    print(textwrap.fill(rec["essay"], width=80, replace_whitespace=False), "\n")


@hydra.main(config_path="../../configs", config_name="infer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    infer_url = f"{cfg.server.url}/v2/models/{cfg.server.model}/infer"

    content = dvc.api.read(cfg.data_path, mode="r")
    df = pd.read_json(StringIO(content), lines=True)
    dataset = df.to_dict(orient="records")

    # dataset = load_dataset(Path(cfg.data_path))
    for rec in random.sample(dataset, cfg.k):
        # ASCII cleaning
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

        payload_text = f'PROMPT: {rec["prompt"]}  ESSAY: {rec["essay"]}'
        pred = query_triton(infer_url, [payload_text])[0]
        pretty_print(rec, pred)


if __name__ == "__main__":
    main()
