import pytorch_lightning as L
import torch
import torch.nn as nn
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        model_name,
        num_classes,
        pretrained=True,
        lr=1e-3,
        weight_decay=1e-5,
        scheduler_patience=3,
        scheduler_factor=0.1,
        min_lr=1e-6
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }
    
        