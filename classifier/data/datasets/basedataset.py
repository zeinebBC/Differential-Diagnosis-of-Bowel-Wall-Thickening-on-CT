# ============================== Imports ==============================

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

from classifier.data.utils.cropping import batch_crop_and_save, crop_nnunet_data
from classifier.data.utils.functions_utils import load_volume, pad_to_shape
from classifier.data.utils.normalizing import process_and_window_dataset
from classifier.data.utils.resampling import batch_resample_and_save

# ============================== Helpers ==============================


def find_existing_file(base_dir, uid, candidates):
    for pattern in candidates:
        path = base_dir / pattern.format(int(uid))
        if path.exists():
            return path
    return None


image_candidates = [
    "{:03d}.b2nd",
    "{:03d}.npz",
    "{:03d}_0000.nii.gz",
    "{:03d}.nii.gz",
    "{}.b2nd",
    "{}.npz",
    "{}_0000.nii.gz",
    "{}.nii.gz",
]

label_candidates = [
    "{:03d}_seg.b2nd",
    "{:03d}_seg.npz",
    "{:03d}_0000_seg.nii.gz",
    "{}.nii.gz",
    "{:03d}_seg.nii.gz",
    "{}_seg.b2nd",
    "{}_seg.npz",
    "{}_0000_seg.nii.gz",
    "{}_seg.nii.gz",
]


# ============================== Base Dataset ==============================


class basedataset(data.Dataset):
    """
    Generic patch-based 3D dataset with:
      - preprocessing
      - train/val/test splits
      - optional full-image return
      - optional label concatenation
    """

    def __init__(
        self,
        patch_size=None,
        dataset_name=None,
        num_patches_per_epoch=None,
        split=None,
        return_full_image=False,
        dual_input=None,
        pp_nnunet_data=None,
        overwrite_preprocessing=False,
        **preprocess_kwargs,
    ):
        self.path_root = Path(os.environ["root"])
        self.dataset_name = dataset_name
        self.pp_nnunet_data = pp_nnunet_data
        self.split = split
        self.return_full_image = return_full_image
        self.dual_input = dual_input
        self.epoch = 0
        self.overwrite_preprocessing = overwrite_preprocessing
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch

        self.splits_file = self.path_root / "raw_data" / dataset_name / "splits.csv"
        self.labels_file = self.path_root / "raw_data" / dataset_name / "labels.csv"

        # ---------------- Paths ----------------
        if split == "test":
            self.images_path = self.path_root / f"pp_data/{dataset_name}/rescaledTs"
            self.labels_path = (
                self.path_root / f"pp_data/{dataset_name}/resampledTs/labels_resampled"
            )
        else:
            if self.pp_nnunet_data:
                root_pp = self.path_root / f"pp_data/{dataset_name}/{pp_nnunet_data}"
                self.images_path = self.labels_path = root_pp / "cropped"
                if not self.images_path.exists():
                    crop_nnunet_data(root_pp, self.images_path)
            else:
                self.images_path = self.path_root / f"pp_data/{dataset_name}/rescaledTr"
                self.labels_path = (
                    self.path_root
                    / f"pp_data/{dataset_name}/resampledTr/labels_resampled"
                )

        need_preprocess = (
            not self.images_path.exists()
            or not self.labels_path.exists()
            or self.overwrite_preprocessing
        )

        if need_preprocess:
            self.preprocess_dataset(**preprocess_kwargs)

        # ---------------- Splits ----------------
        if not self.splits_file.exists():
            self.create_splits()

        df = pd.read_csv(self.splits_file)
        df = df[df["Split"] == split]

        self.images = []
        for _, row in df.iterrows():
            uid = str(row["UID"])
            target = int(row["target"])
            img_path = find_existing_file(self.images_path, uid, image_candidates)
            self.images.append((uid, img_path, target))

        print(f"[BaseDataset] Loaded {len(self.images)} subjects for split='{split}'")

    # ---------------- Basic API ----------------

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_patches_per_epoch or len(self.images)

    def get_item_by_uid(self, uid):
        matches = [item for item in self.images if item[0] == str(uid)]
        if not matches:
            raise ValueError(f"UID {uid} not found in dataset.")
        uid, img_path, target = matches[0]

        img = np.transpose(load_volume(img_path), (2, 1, 0))

        lbl_path = find_existing_file(self.labels_path, uid, label_candidates)
        lbl = (
            np.transpose(load_volume(lbl_path), (2, 1, 0))
            if lbl_path.exists()
            else None
        )

        img_t = torch.from_numpy(img).unsqueeze(0).float()

        if self.dual_input and lbl is not None:
            lbl_t = torch.from_numpy(lbl).unsqueeze(0).float()
            img_t = torch.cat([img_t, lbl_t], dim=0)
        else:
            lbl_t = (
                torch.from_numpy(lbl).unsqueeze(0).float() if lbl is not None else None
            )

        return {
            "uid": uid,
            "source": img_t.unsqueeze(0),
            "label": lbl_t.unsqueeze(0) if lbl_t is not None else None,
            "target": torch.tensor(target, dtype=torch.long),
        }

    # ---------------- Core GetItem ----------------

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        patch_seed = 42 + self.epoch * 100000 + worker_id * 1000 + idx
        rng = np.random.default_rng(patch_seed)

        if self.return_full_image:
            uid, img_path, target = self.images[idx]
        else:
            uid, img_path, target = self.images[rng.integers(0, len(self.images))]

        img = np.transpose(load_volume(img_path), (2, 1, 0))
        lbl = None
        if self.dual_input:
            lbl_path = find_existing_file(self.labels_path, uid, label_candidates)
            if lbl_path.exists():
                lbl = np.transpose(load_volume(lbl_path), (2, 1, 0))

        # Full image mode
        if self.return_full_image:
            print(img.shape, lbl.shape if lbl is not None else None)
            img_t = torch.from_numpy(img).unsqueeze(0).float()

            if lbl is not None:
                lbl_t = torch.from_numpy(lbl).unsqueeze(0).float()
                img_t = torch.cat([img_t, lbl_t], dim=0)
            return {
                "uid": uid,
                "source": img_t,
                "target": torch.tensor(target, dtype=torch.long),
            }

        # Random crop
        D, H, W = img.shape
        ps = self.patch_size

        def get_crop_coords(dim, patch_dim):
            return (
                rng.integers(0, max(1, dim - patch_dim + 1)) if dim > patch_dim else 0
            )

        z, x, y = [get_crop_coords(s, p) for s, p in zip((D, H, W), ps)]

        img_patch = img[z : z + ps[0], x : x + ps[1], y : y + ps[2]]
        lbl_patch = (
            lbl[z : z + ps[0], x : x + ps[1], y : y + ps[2]]
            if lbl is not None
            else None
        )

        if img_patch.shape != ps:
            img_patch = pad_to_shape(img_patch, ps)
            if lbl_patch is not None:
                lbl_patch = pad_to_shape(lbl_patch, ps)

        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()

        if lbl_patch is not None:
            lbl_t = torch.from_numpy(lbl_patch).unsqueeze(0).float()
            img_t = torch.cat([img_t, lbl_t], dim=0)

        return {
            "uid": uid,
            "source": img_t,
            "target": torch.tensor(target, dtype=torch.long),
            "patch": (z, x, y),
        }

    # ---------------- Split & Preprocessing ----------------

    def create_splits(self, val_fraction=0.1, seed=42):
        if self.labels_file is None or not Path(self.labels_file).exists():
            raise FileNotFoundError(
                f"Labels file {self.labels_file} not found. Cannot create splits."
            )
        df = pd.read_csv(self.labels_file)
        df["Split"] = None

        test_path = self.path_root / "raw_data" / self.dataset_name / "imagesTs"

        test_uids = set()
        if test_path.exists():
            nii_files = list(test_path.glob("*.nii.gz"))
            if len(nii_files) > 0:
                test_uids = set()

                for p in nii_files:
                    try:
                        uid = int(p.stem.split("_")[0])
                        test_uids.add(uid)
                    except ValueError:
                        print(f"Skipping invalid UID in filename: {p.name}")

        if test_uids:
            test_mask = df["UID"].isin(test_uids)
            df.loc[test_mask, "Split"] = "test"
        else:
            print(
                "No test set found. Creating train/val split only. "
                "You may need to add test split IDs later for inference."
            )

        trainval = df[~test_mask]
        if len(trainval) == 0:
            print(
                "No training/validation samples found after excluding test set. "
                "Check your labels file and test split."
            )
        else:
            train_idx, val_idx = train_test_split(
                trainval.index,
                test_size=val_fraction,
                stratify=trainval["target"],
                random_state=seed,
            )

            df.loc[train_idx, "Split"] = "train"
            df.loc[val_idx, "Split"] = "val"
        df.to_csv(self.splits_file, index=False)

    def preprocess_dataset(
        self,
        resample_spacing=(1, 1, 1),
        path_data=None,
        use_gt=False,
    ):
        if self.images_path.exists() and self.labels_path.exists():
            # remove images and labels folders
            shutil.rmtree(self.images_path)
            shutil.rmtree(self.labels_path)

        if path_data is None:
            path_data = self.path_root / "raw_data" / self.dataset_name

        suffix = "Ts" if self.split == "test" else "Tr"
        images_dir = path_data / f"images{suffix}"
        if use_gt:
            labels_dir = path_data / f"labels{suffix}"
        else:
            labels_dir = path_data / f"predictions{suffix}"

        cropped = self.path_root / f"pp_data/{self.dataset_name}/raw_cropped{suffix}"
        resampled = self.path_root / f"pp_data/{self.dataset_name}/resampled{suffix}"
        rescaled = self.path_root / f"pp_data/{self.dataset_name}/rescaled{suffix}"

        batch_crop_and_save(
            images_dir,
            labels_dir,
            cropped,
            margin_min=20,
            split=suffix,
            crop_to_colon=False,
        )

        batch_resample_and_save(cropped, resampled, resample_spacing, split=suffix)

        process_and_window_dataset(
            resampled / "images_resampled",
            rescaled,
            window_min=-100,
            window_max=500,
            split=suffix,
        )

        print("Preprocessing complete. Dataset is ready to use.")
