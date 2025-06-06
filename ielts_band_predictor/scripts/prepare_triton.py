from __future__ import annotations

import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=None, version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    src = Path(cfg.src_plan).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source model.plan not found: {src}")

    bert_dir = Path("triton_inference_server/triton_models/bert/1/")
    bert_dir.mkdir(parents=True, exist_ok=True)

    dst = bert_dir / "model.plan"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
        print(f"Existing model.plan removed: {dst}")
    shutil.copy2(src, dst)
    print(f"Copied → {dst}")

    Path("triton_inference_server/triton_models/ielts_pipeline/1/").mkdir(
        parents=True, exist_ok=True
    )
    print("Ensured ensemble directory exists")


if __name__ == "__main__":
    main()
