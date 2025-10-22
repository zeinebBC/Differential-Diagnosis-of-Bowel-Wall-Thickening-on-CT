from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import json
import inspect

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


def _get_qualname(obj: Any) -> str:
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__name__}"
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        optimizer: Union[type, Callable[..., torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None,
        lr_scheduler: Optional[Union[type, Callable[..., torch.optim.lr_scheduler._LRScheduler]]] = None,
        lr_scheduler_kwargs: Dict[str, Any] = None,
        loss: Optional[Union[type, Callable[..., nn.Module]]] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        save_hyperparameters: bool = True,

    ):
        super().__init__()

        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1

        self.optimizer_class = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3, "weight_decay": 1e-2}

        self.lr_scheduler_class = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        self.loss_class = loss or nn.CrossEntropyLoss
        self.loss_kwargs = loss_kwargs or {}
        self.loss_func = self.loss_class(**self.loss_kwargs)

         

    def forward(self, x, cond=None):
        raise NotImplementedError

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError

    def _epoch_end(self, state: str):
        return

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if hasattr(self.trainer.datamodule, "ds_train"):
            dataset = self.trainer.datamodule.ds_train
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if hasattr(self.trainer.datamodule, "ds_val"):
            dataset = self.trainer.datamodule.ds_val
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.current_epoch)

    def training_step(self, batch: dict, batch_idx: int):
        self._step_train += 1
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val)

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")
    
    def on_fit_start(self) -> None:
        return
        #self.log_model_config()
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler_class is not None:
            lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch", "frequency": 1}
            return [optimizer], [lr_scheduler_config]
        return [optimizer]

    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / "best_checkpoint.json", "w") as f:
            json.dump({"best_model_epoch": Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / "best_checkpoint.json", "r") as f:
            path_rel_best_checkpoint = Path(json.load(f)["best_model_epoch"])
        return Path(path_checkpoint_dir) / path_rel_best_checkpoint
    
    @classmethod
    def _get_last_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        checkpoint = Path(path_checkpoint_dir)/ "last.ckpt" 
        if not checkpoint:
            raise FileNotFoundError(f"No last.ckpt found in {path_checkpoint_dir}")
        return checkpoint
    
    @classmethod
    def load_last_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_last_checkpoint = cls._get_last_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_last_checkpoint, **kwargs)
    
    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path: Path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return self.load_weights(checkpoint["state_dict"], **kwargs)

    def load_weights(self, pretrained_weights, strict: bool = True, **kwargs):
        key_filter = kwargs.get("filter", lambda key: key in pretrained_weights)
        init_weights = self.state_dict()
        filtered = {k: v for k, v in pretrained_weights.items() if key_filter(k)}
        init_weights.update(filtered)
        self.load_state_dict(init_weights, strict=strict)
        return self

    def get_model_config(self) -> Dict[str, Any]:
        num_params_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_params_total = sum(p.numel() for p in self.parameters())

        config: Dict[str, Any] = {
            "model": {
                "class": _get_qualname(self.__class__),
                "num_parameters_total": num_params_total,
                "num_parameters_trainable": num_params_trainable,
            },
            "loss": {
                "class": _get_qualname(self.loss_class),
                "kwargs": _to_jsonable(self.loss_kwargs),
            },
            "optimizer": {
                "class": _get_qualname(self.optimizer_class),
                "kwargs": _to_jsonable(self.optimizer_kwargs),
            },
            "lr_scheduler": None,
            "hparams": _to_jsonable(dict(self.hparams)) if hasattr(self, "hparams") else {},
        }

        if self.lr_scheduler_class is not None:
            config["lr_scheduler"] = {
                "class": _get_qualname(self.lr_scheduler_class),
                "kwargs": _to_jsonable(self.lr_scheduler_kwargs),
            }

        try:
            config["model"]["forward_signature"] = str(inspect.signature(self.forward))
        except Exception:
            pass

        return config

    def log_model_config(self, filename: str = "model_config.json") -> None:
        config = self.get_model_config()
        if self.logger is not None:
            as_text = json.dumps(config, indent=2)
            try:
                self.logger.experiment.add_text("model/config", f"<pre>{as_text}</pre>", global_step=0)
            except Exception:
                self.print(as_text)
        log_dir: Optional[str] = None
        try:
            if hasattr(self.trainer, "log_dir") and self.trainer.log_dir:
                log_dir = self.trainer.log_dir
            elif self.logger is not None and hasattr(self.logger, "save_dir") and self.logger.save_dir:
                log_dir = self.logger.save_dir
        except Exception:
            log_dir = None
        try:
            out_dir = Path(log_dir) if log_dir else Path.cwd()
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / filename, "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass


class BasicClassifier(BaseModel):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        spatial_dims: int,
        loss: Optional[Union[type, Callable[..., nn.Module]]] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        optimizer: Union[type, Callable[..., torch.optim.Optimizer]] = torch.optim.SGD,
        optimizer_kwargs: Dict[str, Any] = {'lr': 8e-4, 'momentum': 0.9, 'weight_decay': 1e-2},
        lr_scheduler: Optional[Union[type, Callable[..., torch.optim.lr_scheduler._LRScheduler]]] = torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs: Dict[str, Any] = {"step_size": 15, "gamma": 0.2},
        f1_kwargs: Optional[Dict[str, Any]] = None,
        acc_kwargs: Optional[Dict[str, Any]] = None,
        save_hyperparameters: bool = True,
    ):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.spatial_dims = spatial_dims

        # Decide binary vs multiclass based on out_ch
        self.is_binary = out_ch == 1
        self.task = "binary" if self.is_binary else "multiclass"

        # Choose sensible default losses if not provided
        loss = loss if loss is not None else (nn.BCEWithLogitsLoss if self.is_binary else nn.CrossEntropyLoss)
        loss_kwargs = {"pos_weight": torch.tensor([0.4 / 0.6]) }  if self.is_binary else {"weight": torch.tensor([0.4, 0.6], dtype=torch.float)}
        

        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss=loss,
            loss_kwargs=loss_kwargs,
            save_hyperparameters=save_hyperparameters,
        )

        # Metrics
        self.f1_kwargs = dict(f1_kwargs or {})
        self.acc_kwargs = dict(acc_kwargs or {})
        if self.is_binary:
            # torchmetrics will threshold probabilities at 0.5
            self.f1 = nn.ModuleDict({state: F1Score(task="binary", **self.f1_kwargs) for state in ["train_", "val_"]})
            self.acc = nn.ModuleDict({state: Accuracy(task="binary", **self.acc_kwargs) for state in ["train_", "val_"]})
        else:
            self.f1_kwargs.update({"num_classes": out_ch})
            self.acc_kwargs.update({"num_classes": out_ch})
            self.f1 = nn.ModuleDict({state: F1Score(task="multiclass", **self.f1_kwargs) for state in ["train_", "val_"]})
            self.acc = nn.ModuleDict({state: Accuracy(task="multiclass", **self.acc_kwargs) for state in ["train_", "val_"]})
        
        
            
       

    def compute_loss(self, pred, target):
        return self.loss_func(pred, target)

    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        target = batch["target"]
        batch_size = target.shape[0]
        self.batch_size = batch_size

        pred = self(**batch)

        if self.is_binary:
            # BCEWithLogitsLoss expects float targets; match shape
            pred_logits = pred.squeeze(-1)
            loss_val = self.compute_loss(pred_logits, target.float())
            with torch.no_grad():
                probs = torch.sigmoid(pred_logits)
                self.acc[state + "_"].update(probs, target.int())
                self.f1[state + "_"].update(probs, target.int())
        else:
            # CrossEntropyLoss expects class indices (long)
            loss_val = self.compute_loss(pred, target.long())
            with torch.no_grad():
                self.acc[state + "_"].update(pred, target)
                self.f1[state + "_"].update(pred, target)

        self.log(
            f"{state}/loss",
            loss_val,
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        return loss_val

    def _epoch_end(self, state: str):
        for name, metric in [("ACC", self.acc[state + "_"]), ("F1Score", self.f1[state + "_"])]:
            self.log(
                f"{state}/{name}",
                metric.compute(),
                batch_size=getattr(self, "batch_size", None) or 1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            metric.reset()


    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.logger is not None:
            extra_hparams = {}
            for k, v in self.f1_kwargs.items():
                extra_hparams[f"f1_{k}"] = str(v)
            for k, v in self.acc_kwargs.items():
                extra_hparams[f"acc_{k}"] = str(v)
            self.logger.log_hyperparams(extra_hparams)

