# scripts/train.py
"""
Fine-tune BERT on an IELTS-essay dataset with Lightning + Hydra.

• All parameters (model, datamodule, trainer, callbacks, etc.)
  come from YAML files in configs/.
• No external config-schema (dataclasses) is used.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import hydra
import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------
    # 1. Show the final, fully-resolved config for debugging
    # ------------------------------------------------------------------
    print("\n╭─ Final config ─" + "─" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("╰" + "─" * 75 + "\n")

    # ------------------------------------------------------------------
    # 2. Reproducibility
    # ------------------------------------------------------------------
    pl.seed_everything(cfg.seed, workers=True)

    # ------------------------------------------------------------------
    # 3. Instantiate DataModule, Model, Callbacks, Optimizer
    #    (Hydra's `instantiate` injects kwargs from YAML)
    # ------------------------------------------------------------------
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    # Optimizer often needs model.parameters()
    # print(cfg.optimizer)
    # print(cfg.callback_early_stop)
    # print(cfg.callback_lr_monitor)
    # optimizer = instantiate(cfg.optimizer, params=model.parameters())

    callbacks: List[pl.callbacks.Callback] = [instantiate(cfg.callback_lr_monitor)]

    # Add a default checkpoint callback if user didn't specify one
    if not any(isinstance(c, ModelCheckpoint) for c in callbacks):
        ckpt_dir = Path("artifacts") / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-{step}-{val_mae:.3f}",
                monitor="val_mae",
                mode="min",
                save_top_k=1,
            )
        )

    # ------------------------------------------------------------------
    # 4. MLflow logger (tracking URI from env or default ./mlruns)
    # ------------------------------------------------------------------
    # os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    logger_cfg = cfg.get("logger")

    if logger_cfg is not None:
        # Hydra-style: превращаем DictConfig -> объект
        logger = instantiate(logger_cfg)
    else:
        logger = False  # Lightning принимает bool = отключить логи

    # logger = MLFlowLogger(
    #     experiment_name="ielts-band-predictor",
    #     tracking_uri=mlflow_uri,
    #     tags={"run_type": "train"},
    # )

    # ------------------------------------------------------------------
    # 5. Build Lightning Trainer from the YAML block
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )

    # ------------------------------------------------------------------
    # 6. Fit!
    # ------------------------------------------------------------------
    trainer.fit(model=model, datamodule=datamodule)

    # Optional: evaluate on the validation set after training
    # trainer.validate(model=model, datamodule=datamodule, verbose=True)

    ckpt_cb = next(
        (cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)), None
    )

    if ckpt_cb and ckpt_cb.best_model_path:
        src = Path(ckpt_cb.best_model_path)
        dst = src.with_name("best.ckpt")

        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.name)  # relative link
            print(f"🔗  best.ckpt → {src.name}")
        except (OSError, NotImplementedError):
            # Windows without dev mode: fall back to copy
            shutil.copy2(src, dst)
            print(f"📋 Copied {src.name} → best.ckpt (symlinks unsupported)")
    else:
        print("⚠️  No best_model_path found ― did training finish?")


if __name__ == "__main__":
    main()
