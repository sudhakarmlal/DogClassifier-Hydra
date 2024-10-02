import os
from utils import logging_utils

# Setup root directory
from pathlib import Path
#log_file = Path(__file__).parent.parent / "logs" / "train.log"
#logging_utils.setup_logger(log_file)
#root = Path(__file__).parent.parent

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback
import rootutils
from utils.logging_utils import setup_logger, task_wrapper, logger, log_metrics_table


# Setup Python path
#os.environ["PYTHONPATH"] = str(root)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def instantiate_callbacks(callback_cfg: DictConfig) -> list[Callback]:
    callbacks: list[Callback] = []
    if callback_cfg:
        for _, cb_conf in callback_cfg.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    loggers: list[Logger] = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                logger.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up logger
    log=logging_utils.setup_logger(Path(cfg.paths.output_dir) / "train.log")
    #log = logging_utils.get_logger(__name__)

    # Set seed for reproducibility
    if cfg.get("seed"):
        logger.info(f"Setting seed: {cfg.seed}")
        #logging_utils.set_seed(cfg.seed)

    # Create datamodule
    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    print(cfg.data)
    datamodule = hydra.utils.instantiate(cfg.data)

    # Create model
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    print(cfg.model)
    model = hydra.utils.instantiate(cfg.model)

    # Create callbacks
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    # Create loggers
    loggers = instantiate_loggers(cfg.get("logger"))

    # Create trainer
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
    )

    # Train the model
    if cfg.get("train"):
        logger.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        logger.info("Training completed!")
        logger.info(f"Train metrics:\n{trainer.callback_metrics}")

    # Evaluate model on test set after training
    if cfg.get("test"):
        logger.info("Starting testing!")
        best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            logger.info("No best model checkpoint found. Using current model weights.")
            trainer.test(model=model, datamodule=datamodule)
        logger.info("Testing completed!")
        logger.info(f"Test metrics:\n{trainer.callback_metrics}")

    # Make sure everything closed properly
    logger.info("Finalizing!")
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