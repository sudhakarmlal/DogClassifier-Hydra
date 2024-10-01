import os
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback

from src.utils import logging_utils

# Setup Python path
os.environ["PYTHONPATH"] = str(root)

def instantiate_callbacks(callback_cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []
    if callback_cfg:
        for _, cb_conf in callback_cfg.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    loggers: list[Logger] = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up logger
    log = logging_utils.get_logger(__name__)

    # Set seed for reproducibility
    if cfg.get("seed"):
        log.info(f"Setting seed: {cfg.seed}")
        logging_utils.set_seed(cfg.seed)

    # Create datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Create model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Create callbacks
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # Create loggers
    loggers = instantiate_loggers(cfg.get("logger"))

    # Create trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Train the model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Training completed!")
        log.info(f"Train metrics:\n{trainer.callback_metrics}")

    # Evaluate model on test set after training
    if cfg.get("test"):
        log.info("Starting testing!")
        best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
        if best_model_path and os.path.exists(best_model_path):
            log.info(f"Loading best model from {best_model_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            log.info("No best model checkpoint found. Using current model weights.")
            trainer.test(model=model, datamodule=datamodule)
        log.info("Testing completed!")
        log.info(f"Test metrics:\n{trainer.callback_metrics}")

    # Make sure everything closed properly
    log.info("Finalizing!")
    logging_utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=loggers,
    )

if __name__ == "__main__":
    main()