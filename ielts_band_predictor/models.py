from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)


class BertBandRegressor(pl.LightningModule):
    def __init__(
        self,
        pretrained_name: str = "bert-base-uncased",
        lr: float = 2e-5,
        weight_decay: float = 1e-2,
        freeze_n_layers: int = 8,
        warmup_steps: int | None = None,
        total_steps: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # backbone
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size

        # simple linear head
        self.reg_head = nn.Linear(hidden, 1)

        # freeze bottom-N layers
        if freeze_n_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_n_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        # metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden]
        logits = self.reg_head(pooled).squeeze(-1)  # [batch]
        return logits

    def training_step(self, batch, _):
        y_hat = self(**batch)
        y_true = batch["labels"].float()
        loss = F.mse_loss(y_hat, y_true)

        mae = torch.mean(torch.abs(y_hat - y_true))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mae", mae, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        y_hat = self(**batch)
        y_true = batch["labels"].float()
        loss = F.mse_loss(y_hat, y_true)
        mae = torch.mean(torch.abs(y_hat - y_true))

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        if self.hparams.total_steps and self.hparams.warmup_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "name": "linear_warmup",
                },
            }
        return optimizer
