import json
import pathlib
import random
import textwrap
from typing import List

import requests

from ielts_band_predictor.scripts.remove_nonascii import strip_non_ascii

SERVER_URL = "http://localhost:8010"
MODEL_NAME = "ielts_pipeline"
INFER_URL = f"{SERVER_URL}/v2/models/{MODEL_NAME}/infer"


def _payload(essays: List[str]) -> bytes:
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


def query_triton(essays: List[str]) -> List[float]:
    try:
        resp = requests.post(
            INFER_URL, data=_payload(essays), headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError("Triton request failed:\n" + resp.text) from e

    obj = resp.json()
    scores = obj["outputs"][0]["data"]
    return scores


DATA_PATH = pathlib.Path("data/raw/essays.jsonl")


def load_dataset(path: pathlib.Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def sample_essays(dataset, k=1):
    """Return k random records (list of dicts)."""
    return random.sample(dataset, k)


def pretty_print(record, pred):
    print("=" * 80)
    print(f"BAND  : {record['band']} PREDICTED : {pred:.2f}")
    print(f"PROMPT: {record['prompt']}")
    print("\nESSAY :\n")
    print(textwrap.fill(record["essay"], width=80, replace_whitespace=False))
    print()


if __name__ == "__main__":
    dataset = load_dataset(DATA_PATH)
    for rec in sample_essays(dataset, 3):
        rec["prompt"] = strip_non_ascii(rec["prompt"], ratio=0.05, absolute=50)
        rec["essay"] = strip_non_ascii(rec["essay"], ratio=0.05, absolute=50)
        if not rec["prompt"] or not rec["essay"]:
            print("essay or prompt contain too much non-ascii symbols, skiping it")
            continue

        predicted = query_triton([f'PROMPT: {rec["prompt"]}  ESSAY: {rec["essay"]}'])[0]
        pretty_print(rec, predicted)
