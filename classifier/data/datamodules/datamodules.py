
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp 



class DataModuleCC(pl.LightningDataModule):

    def __init__(self,
                 ds_train=None,
                 ds_val=None,
                 ds_test=None,
                 batch_size=1,
                 batch_size_val=None,
                 batch_size_test=None,
                 num_workers=mp.cpu_count(),
                 seed=42,
                 pin_memory=False):
        super().__init__()
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.batch_size = batch_size
        self.batch_size_val = batch_size if batch_size_val is None else batch_size_val
        self.batch_size_test = batch_size if batch_size_test is None else batch_size_test

        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory

    # ----------------------------
    # Training loader
    # ----------------------------
    def train_dataloader(self):
        if self.ds_train is None:
            raise AssertionError("A training set was not initialized.")
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,               
            drop_last=True,
            pin_memory=self.pin_memory
        )

    # ----------------------------
    # Validation loader
    # ----------------------------
    def val_dataloader(self):
        if self.ds_val is None:
            raise AssertionError("A validation set was not initialized.")
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory
        )

    # ----------------------------
    # Test loader
    # ----------------------------
    def test_dataloader(self):
        if self.ds_test is None:
            raise AssertionError("A test set was not initialized.")
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory
        )
