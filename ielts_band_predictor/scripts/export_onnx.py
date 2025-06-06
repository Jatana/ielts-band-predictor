from __future__ import annotations

import hydra
import onnx
import onnxsim
import torch
from omegaconf import DictConfig, OmegaConf

from ielts_band_predictor.models import BertBandRegressor


@hydra.main(config_path="../../configs", config_name="export_onnx", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    model = BertBandRegressor.load_from_checkpoint(cfg.ckpt).eval().to("cpu")

    ids = torch.zeros(1, cfg.max_len, dtype=torch.long)
    mask = torch.ones_like(ids)

    torch.onnx.export(
        model,
        (ids, mask),
        cfg.onnx,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
    )

    onnx_model, _ = onnxsim.simplify(onnx.load(cfg.onnx))
    onnx.save(onnx_model, cfg.onnx)
    print(f"ONNX saved → {cfg.onnx}")


if __name__ == "__main__":
    main()
