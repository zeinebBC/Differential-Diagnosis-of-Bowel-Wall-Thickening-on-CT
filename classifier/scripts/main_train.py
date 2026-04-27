import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from classifier.data import DataModuleCC, basedataset
from classifier.models import ResNet


# ------------------------------------------
# Helper: get_model
# ------------------------------------------
def get_model(cfg):
    if cfg.model.type == "ResNet":
        return ResNet(
            in_ch=cfg.model.in_ch,
            out_ch=cfg.model.out_ch,
            optimizer=getattr(torch.optim, cfg.optimizer.type.split(".")[-1]),
            optimizer_kwargs=cfg.optimizer.kwargs,
            lr_scheduler=getattr(
                torch.optim.lr_scheduler, cfg.lr_scheduler.type.split(".")[-1]
            ),
            lr_scheduler_kwargs=cfg.lr_scheduler.kwargs,
            model=cfg.model.resnet_model,
            pretrained=cfg.model.pretrained,
            kwargs_resnet=cfg.model.kwargs_resnet,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.type}")


# ------------------------------------------
# Main
# ------------------------------------------
def main():
    # -------------------- Load config --------------------
    config_file = Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    cfg.training = SimpleNamespace(**cfg.training)
    cfg.model = SimpleNamespace(**cfg.model)
    cfg.optimizer = SimpleNamespace(**cfg.optimizer)
    cfg.lr_scheduler = SimpleNamespace(**cfg.lr_scheduler)

    # -------------------- Paths --------------------
    path_root = Path(os.environ.get("root", "."))
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = (
        path_root
        / cfg.training.output_dir
        / f"{cfg.model.type}_{cfg.training.dataset}_{current_time}"
    )
    path_run_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run folder
    (path_run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=4))

    # -------------------- Device --------------------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    # -------------------- Datasets --------------------
    dataset_kwargs = dict(
        dataset_name=cfg.training.dataset,
        patch_size=cfg.training.patch_size,
        use_gt=cfg.training.use_gt,
        dual_input=cfg.training.dual_input,
        pp_nnunet_data=cfg.training.pp_nnunet_data,
        resample_spacing=tuple(cfg.training.resample_spacing),
        path_data=cfg.training.path_data,
    )

    ds_train = basedataset(
        **dataset_kwargs,
        overwrite_preprocessing=cfg.training.overwrite_preprocessing,
        num_patches_per_epoch=cfg.training.num_patches_per_epoch_train,
        split="train",
    )

    ds_val = basedataset(
        **dataset_kwargs,
        overwrite_preprocessing=False,
        num_patches_per_epoch=cfg.training.num_patches_per_epoch_val,
        split="val",
    )

    # -------------------- DataModule --------------------
    dm = DataModuleCC(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=cfg.training.batch_size,
        pin_memory=True,
        shuffle=True,  #########
        num_workers=cfg.training.num_workers,
    )

    # -------------------- Model --------------------
    model = get_model(cfg)

    # -------------------- Logging and Callbacks --------------------
    monitor_metric = "val_ACC"
    mode = "max"
    path_ml_dir = path_root / path_run_dir.parent.parent / "mlruns"
    print(path_ml_dir)
    logger = MLFlowLogger(
        experiment_name=f"Classifier_{cfg.training.dataset}",
        tracking_uri=f"file:{path_ml_dir}",
        run_name=f"{cfg.model.type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(path_run_dir),
        filename="best",
        monitor="val_ACC",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor=monitor_metric, min_delta=0.0, patience=50, mode=mode),
        checkpoint_cb,
    ]

    # -------------------- Trainer --------------------
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        strategy="auto",
        precision="16-mixed",
        default_root_dir=str(path_run_dir),
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        max_epochs=cfg.training.num_epochs,
        num_sanity_val_steps=2,
        logger=logger,
    )

    # -------------------- Train --------------------
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.training.resume_from_checkpoint)

    # -------------------- Save Best Model --------------------


if __name__ == "__main__":
    main()
