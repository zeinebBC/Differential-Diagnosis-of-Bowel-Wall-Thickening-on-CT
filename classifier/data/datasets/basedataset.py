# ============================== Imports ==============================

import json
import os
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
def generate_target_mapping_method(self):
    DIV_CASES = Path(
        os.environ.get(
            "CC_DIV_CASES", "/data/colon_cancer/Classifier/filename_mapping.json"
        )
    )
    if not DIV_CASES.exists():
        raise FileNotFoundError(f"Mapping file {DIV_CASES} not found.")

    with open(DIV_CASES) as f:
        known_uids = json.load(f)
    known_uids_set = set(known_uids.keys())

    # Load existing labels if any
    existing_df = (
        pd.read_csv(self.labels_file) if self.labels_file.exists() else pd.DataFrame()
    )
    existing_uids = (
        set(existing_df["UID"].astype(str)) if not existing_df.empty else set()
    )

    mapping = []
    num_div, num_cancer = 0, 0

    # Loop over both train and test folders
    for split in ["Tr", "Ts"]:
        images_folder = self.path_root / f"raw_data/{self.dataset_name}/images{split}"
        images = list(images_folder.glob("*.nii*"))

        for img_path in images:
            uid = img_path.stem.replace("_0000.nii", "")
            if uid in existing_uids:
                continue

            target = 0 if uid in known_uids_set else 1
            num_div += target == 0
            num_cancer += target == 1

            mapping.append({"UID": uid, "img_path": img_path, "target": target})

    columns_order = ["UID", "img_path", "target"]
    new_df = (
        pd.DataFrame(mapping)[columns_order]
        if mapping
        else pd.DataFrame(columns=columns_order)
    )
    final_df = (
        pd.concat([existing_df, new_df], ignore_index=True)
        if not existing_df.empty
        else new_df
    )
    final_df.to_csv(self.labels_file, index=False)

    print(f"Saved classification labels for {len(final_df)} images")
    print(f"Diverticulitis: {num_div}, Colon cancer: {num_cancer}")


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
        transforms=None,
        num_patches_per_epoch=None,
        split=None,
        return_full_image=False,
        use_labels=None,
        pp_nnunet_data=None,
        **preprocess_kwargs,
    ):
        self.path_root = Path("/data/colon_cancer/CC_Detection")
        self.dataset_name = dataset_name
        self.pp_nnunet_data = pp_nnunet_data
        self.split = split
        self.return_full_image = return_full_image
        self.use_labels = use_labels
        self.epoch = 0

        self.patch_size = patch_size
        self.transforms = transforms
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

        overwrite_flags = [
            "overwrite_cropping",
            "overwrite_resample",
            "overwrite_window",
        ]
        need_preprocess = (
            not self.images_path.exists()
            or not self.labels_path.exists()
            or any(preprocess_kwargs.get(f, False) for f in overwrite_flags)
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
        if self.transforms:
            img_t = self.transforms(img_t)

        if self.use_labels and lbl is not None:
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
        if self.use_labels:
            lbl_path = find_existing_file(self.labels_path, uid, label_candidates)
            if lbl_path.exists():
                lbl = np.transpose(load_volume(lbl_path), (2, 1, 0))

        # Full image mode
        if self.return_full_image:
            img_t = torch.from_numpy(img).unsqueeze(0).float()
            if self.transforms:
                img_t = self.transforms(img_t)
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
        if self.transforms:
            img_t = self.transforms(img_t)
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
        df = pd.read_csv(self.labels_file)
        df["Split"] = None

        test_mask = df["img_path"].str.contains("imagesTs")
        df.loc[test_mask, "Split"] = "test"

        trainval = df[~test_mask]
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
        overwrite_cropping=True,
        overwrite_resample=True,
        overwrite_window=True,
        resample_spacing=(1, 1, 1),
        path_data=None,
        use_gt=True,
    ):
        if path_data is None:
            path_data = self.path_root / "raw_data" / self.dataset_name

        suffix = "Ts" if self.split == "test" else "Tr"
        images_dir = path_data / f"images{suffix}"
        labels_dir = path_data / f"labels{suffix}"

        cropped = self.path_root / f"pp_data/{self.dataset_name}/raw_cropped{suffix}"
        resampled = self.path_root / f"pp_data/{self.dataset_name}/resampled{suffix}"
        rescaled = self.path_root / f"pp_data/{self.dataset_name}/rescaled{suffix}"

        if overwrite_cropping or not cropped.exists():
            batch_crop_and_save(
                images_dir,
                labels_dir,
                cropped,
                margin_min=20,
                split=suffix,
                crop_to_colon=True,
            )

        if overwrite_resample or not resampled.exists():
            batch_resample_and_save(cropped, resampled, resample_spacing, split=suffix)

        if overwrite_window or not rescaled.exists():
            process_and_window_dataset(
                resampled / "images_resampled",
                rescaled,
                window_min=-100,
                window_max=500,
                split=suffix,
            )

        print("Preprocessing complete. Dataset is ready to use.")
