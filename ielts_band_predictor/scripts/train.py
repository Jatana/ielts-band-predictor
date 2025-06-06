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
    print("\n╭─ Final config ─" + "─" * 60)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("╰" + "─" * 75 + "\n")

    pl.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    callbacks: List[pl.callbacks.Callback] = [instantiate(cfg.callback_lr_monitor)]

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

    logger_cfg = cfg.get("logger")

    if logger_cfg is not None:
        logger = instantiate(logger_cfg)
    else:
        logger = False

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **OmegaConf.to_container(cfg.trainer, resolve=True),
    )

    trainer.fit(model=model, datamodule=datamodule)

    ckpt_cb = next(
        (cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)), None
    )

    if ckpt_cb and ckpt_cb.best_model_path:
        src = Path(ckpt_cb.best_model_path)
        dst = src.with_name("best.ckpt")

        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.name)
            print(f"best.ckpt → {src.name}")
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
            print(f"Copied {src.name} → best.ckpt (symlinks unsupported)")
    else:
        print("No best_model_path found")


if __name__ == "__main__":
    main()
