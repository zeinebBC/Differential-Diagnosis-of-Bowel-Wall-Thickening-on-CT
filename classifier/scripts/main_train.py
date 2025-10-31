
from pathlib import Path
import argparse
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import json 
from classifier.data import DataModuleCC  
from classifier.models import ResNet 
from classifier.data import ColonCancer
import os
from classifier.scripts.utils.functions import str2bool


# ------------------------------------------
# Helper: get_model
# ------------------------------------------
def get_model(config):
    if config["model"]["type"] == 'ResNet':

        return ResNet(
            in_ch=config["model"]["in_ch"],
            out_ch=config["model"]["out_ch"],
            optimizer=getattr(torch.optim, config["optimizer"]["type"].split(".")[-1]),
            optimizer_kwargs=config["optimizer"]["kwargs"],
            lr_scheduler=getattr(torch.optim.lr_scheduler, config["lr_scheduler"]["type"].split(".")[-1]),
            lr_scheduler_kwargs=config["lr_scheduler"]["kwargs"],
            model=config["model"].get("resnet_model", 18),
            pretrained=config["model"].get("pretrained", False),
            kwargs_resnet=config["model"].get("kwargs_resnet", {}),
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['type']}")
    

# ------------------------------------------
# Main
# ------------------------------------------

def main():
    
    #read from a config_file 
    config_file =Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ColonCancer', choices=['ColonCancer']) 
    parser.add_argument('--path_root', type=str, default='/data/colon_cancer')
    parser.add_argument('--labels_file', type=str, default=None)
    parser.add_argument('--splits_file', type=str, default=None)
    parser.add_argument('--cross_val', type=str2bool, default=False)
    parser.add_argument('--resample_spacing', nargs=3, type=float, default=[0.7, 0.7, 0.8])
    parser.add_argument('--div_cases', type=str, default=None)
    parser.add_argument("--transforms", type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1000)

    # fixed boolean arguments
    parser.add_argument('--overwrite_cropping', type=str2bool, default=True)
    parser.add_argument('--overwrite_resample', type=str2bool, default=True)
    parser.add_argument('--overwrite_window', type=str2bool, default=True)

    parser.add_argument('--model', type=str, required=True, choices=['ResNet'])
    parser.add_argument('--path_root_output', type=str, default='./runs', help="Root output path")
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_patches_per_epoch_train', type=int, default=2500)
    parser.add_argument('--num_patches_per_epoch_val', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--patch_size', nargs=3, type=int, default=[124, 124, 32])
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    """
    

    # ------------ Output setup ------------
    path_root=  os.environ.get("root_path")
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(path_root) / Path(config["training"]["output_dir"]) / f'logs/{config["model"]["type"]}_{config["training"]["dataset"]}_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    config_save_path = path_run_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    
   
    # ------------ Load Data ------------


    
    ds_train = ColonCancer(
        dataset_name=config["training"]["dataset"],
        patch_size=config["training"]["patch_size"],
        transforms=config["training"]["transforms"],
        num_patches_per_epoch=config["training"]["num_patches_per_epoch_train"],
        #fold=config["training"]["fold"],
        split='train',
        use_labels=config["training"]["use_labels"],
        overwrite_cropping=config["training"]["overwrite_cropping"],
        overwrite_resample=config["training"]["overwrite_resample"],
        overwrite_window=config["training"]["overwrite_window"],
        resample_spacing=tuple(config["training"]["resample_spacing"]),
        pp_nnunet_data=config["training"]["pp_nnunet_data"]
        #cross_val=config["training"]["cross_val"],
        #labels_path = f"/data/colon_cancer/Classifier/resampledTr/labels_resampled"
       
        
    )

    ds_val = ColonCancer(
        dataset_name=config["training"]["dataset"],
        patch_size=config["training"]["patch_size"],
        transforms=config["training"]["transforms"],
        num_patches_per_epoch=config["training"]["num_patches_per_epoch_val"],
        #fold=config["training"]["fold"],
        split='val',
        use_labels=config["training"]["use_labels"],
        pp_nnunet_data=config["training"]["pp_nnunet_data"]
        
        #labels_path = f"/data/colon_cancer/Classifier/resampledTr/labels_resampled"
        
    )



    # ------------ DataModule ------------
    dm = DataModuleCC(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=config["training"]["batch_size"],
        pin_memory=True,
        num_workers=config["training"]["num_workers"],
        
    )

    # ------------ Initialize Model ------------
    model = get_model(config)
    
    # ------------ Logging and Callbacks ------------
    to_monitor = "val/ACC"
    min_max = "max"
    path_ml_dir = Path(path_root) / Path(config["training"]["output_dir"]) / f'mlruns'

    logger = MLFlowLogger(
    experiment_name=f"Classifier_{config['training']['dataset']}",
    tracking_uri=f"file:{path_ml_dir}",
    run_name=f"{config['model']['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor=to_monitor,min_delta=0.0, patience=50, mode=min_max)
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        filename="last_epoch",       
        monitor=to_monitor,
        save_top_k=1,
        mode=min_max,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )

    # ------------ Trainer ------------
    trainer = Trainer(
        accelerator=accelerator,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor, early_stopping],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        max_epochs=config["training"]["num_epochs"],
        num_sanity_val_steps=2,
        logger=logger,
    )

    # ------------ Train ------------
    trainer.fit(model, datamodule=dm, ckpt_path=config["training"]["resume_from_checkpoint"])

    # ------------ Save Best Model Path ------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)
    
if __name__ == "__main__":
    main()