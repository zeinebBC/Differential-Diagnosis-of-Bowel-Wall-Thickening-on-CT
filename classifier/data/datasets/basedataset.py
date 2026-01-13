# ============================== Imports ==============================

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from scipy import ndimage

from classifier.data.utils.cropping import batch_crop_and_save, crop_nnunet_data
from classifier.data.utils.resampling import batch_resample_and_save
from classifier.data.utils.normalizing import process_and_window_dataset
from classifier.data.utils.functions_utils import load_volume, pad_to_shape

import shutil 
# ============================== Helpers ==============================
def generate_target_mapping_method(self):
    DIV_CASES = Path(os.environ.get("CC_DIV_CASES", "/data/colon_cancer/Classifier/filename_mapping.json"))
    if not DIV_CASES.exists():
        raise FileNotFoundError(f"Mapping file {DIV_CASES} not found.")
    
    with open(DIV_CASES) as f:
        known_uids = json.load(f)
    known_uids_set = set(known_uids.keys())

    # Load existing labels if any
    existing_df = pd.read_csv(self.labels_file) if self.labels_file.exists() else pd.DataFrame()
    existing_uids = set(existing_df['UID'].astype(str)) if not existing_df.empty else set()

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
    new_df = pd.DataFrame(mapping)[columns_order] if mapping else pd.DataFrame(columns=columns_order)
    final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
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
    "{:03d}.b2nd", "{:03d}.npz", "{:03d}_0000.nii.gz", "{:03d}.nii.gz",
    "{}.b2nd", "{}.npz", "{}_0000.nii.gz", "{}.nii.gz",
]

label_candidates = [
    "{:03d}_seg.b2nd", "{:03d}_seg.npz", "{:03d}_0000_seg.nii.gz",
    "{}.nii.gz", "{:03d}_seg.nii.gz", "{}_seg.b2nd", "{}_seg.npz",
    "{}_0000_seg.nii.gz", "{}_seg.nii.gz",
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
            self.labels_path = self.path_root / f"pp_data/{dataset_name}/resampledTs/labels_resampled"
        else:
            if self.pp_nnunet_data:
                root_pp = self.path_root / f"pp_data/{dataset_name}/{pp_nnunet_data}"
                self.images_path = self.labels_path = root_pp / "cropped"
                if not self.images_path.exists():
                    crop_nnunet_data(root_pp, self.images_path)
            else:
                self.images_path = self.path_root / f"pp_data/{dataset_name}/rescaledTr"
                self.labels_path = self.path_root / f"pp_data/{dataset_name}/resampledTr/labels_resampled"

        overwrite_flags = ["overwrite_cropping", "overwrite_resample", "overwrite_window"]
        need_preprocess = (
            not self.images_path.exists() or not self.labels_path.exists()
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
        lbl = np.transpose(load_volume(lbl_path), (2, 1, 0)) if lbl_path.exists() else None

        img_t = torch.from_numpy(img).unsqueeze(0).float()
        if self.transforms:
            img_t = self.transforms(img_t)

        if self.use_labels and lbl is not None:
            lbl_t = torch.from_numpy(lbl).unsqueeze(0).float()
            img_t = torch.cat([img_t, lbl_t], dim=0)
        else:
            lbl_t = torch.from_numpy(lbl).unsqueeze(0).float() if lbl is not None else None

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
            return {"uid": uid, "source": img_t, "target": torch.tensor(target, dtype=torch.long)}

        # Random crop
        D, H, W = img.shape
        ps = self.patch_size
        def get_crop_coords(dim, patch_dim):
            return rng.integers(0, max(1, dim - patch_dim + 1)) if dim > patch_dim else 0
        z, x, y = [get_crop_coords(s, p) for s, p in zip((D, H, W), ps)]

        img_patch = img[z:z+ps[0], x:x+ps[1], y:y+ps[2]]
        lbl_patch = lbl[z:z+ps[0], x:x+ps[1], y:y+ps[2]] if lbl is not None else None

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

        return {"uid": uid, "source": img_t, "target": torch.tensor(target, dtype=torch.long), "patch": (z, x, y)}

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
            batch_crop_and_save(images_dir, labels_dir, cropped, margin_min=20, split=suffix, crop_to_colon=True)

        if overwrite_resample or not resampled.exists():
            batch_resample_and_save(cropped, resampled, resample_spacing, split=suffix)

        if overwrite_window or not rescaled.exists():
            process_and_window_dataset(
                resampled / "images_resampled",
                rescaled,
                window_min=-100,
                window_max=500,
                split=suffix
            )
        
        print("Preprocessing complete. Dataset is ready to use.")


# ============================== Colon Cancer Dataset ==============================

class ColonCancer_center(basedataset):
    """
    Colon Cancer dataset that uses ALL centered, overlapping patches
    covering the ROI of each image.

    One dataset item == one patch
    """

    def __init__(
        self,
        patch_size,
        over_frac=0.25,
        transforms=None,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            transforms=transforms,
            **kwargs,
        )

        self.patch_size = np.array(patch_size)
        self.over_frac = over_frac

        # --------------------------------------------------
        # Build list of ALL patch centers (metadata only)
        # --------------------------------------------------
        self.samples = []
        self._build_patch_index()

    # ==================================================
    # ---------------- Patch utilities -----------------
    # ==================================================

    def _pad_patch(self, patch):
        pad = []
        for i in range(3):
            diff = self.patch_size[i] - patch.shape[i]
            pad.append((diff // 2, diff - diff // 2))
        return np.pad(patch, pad, mode="constant")

    def _extract_patch_at_center(self, vol, center):
        start = center - self.patch_size // 2
        end = start + self.patch_size

        slices = []
        for d in range(3):
            s = max(0, start[d])
            e = min(vol.shape[d], end[d])
            slices.append(slice(s, e))

        patch = vol[tuple(slices)]

        if patch.shape != tuple(self.patch_size):
            patch = self._pad_patch(patch)

        return patch

    # ==================================================
    # -------- Centered overlapping centers ------------
    # ==================================================

    def _centered_centers(self, min_c, max_c, patch, stride):
        length = max_c - min_c

        if length <= patch:
            return [(min_c + max_c) // 2]

        n = int(np.ceil((length - patch) / stride)) + 1
        total = (n - 1) * stride + patch
        offset = (total - length) // 2

        return [
            min_c - offset + i * stride + patch // 2
            for i in range(n)
        ]

    def _extract_centered_covering_centers(self, roi_mask):
        coords = np.argwhere(roi_mask)
        if len(coords) == 0:
            return []

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0) + 1

        stride = np.maximum(
            (self.patch_size * (1 - self.over_frac)).astype(int),
            1,
        )

        cz = self._centered_centers(zmin, zmax, self.patch_size[0], stride[0])
        cy = self._centered_centers(ymin, ymax, self.patch_size[1], stride[1])
        cx = self._centered_centers(xmin, xmax, self.patch_size[2], stride[2])

        centers = []
        for z in cz:
            for y in cy:
                for x in cx:
                    centers.append(np.array([z, y, x]))

        return centers

    # ==================================================
    # -------- Build dataset index ---------------------
    # ==================================================

    def _build_patch_index(self):
        print("Building centered patch index...")

        for uid, img_path, class_target in self.images:
            img = np.transpose(load_volume(img_path), (2, 1, 0))

            seg_path = find_existing_file(
                self.labels_path, uid, label_candidates
            )
            seg = np.transpose(load_volume(seg_path), (2, 1, 0))

            roi_mask = seg > 0

            centers = self._extract_centered_covering_centers(roi_mask)

            if len(centers) == 0:
                continue

            for c in centers:
                self.samples.append(
                    (uid, img_path, class_target, c)
                )

        if len(self.samples) == 0:
            raise RuntimeError("No patches found in dataset")

        print(f"Total patches: {len(self.samples)}")

    # ==================================================
    # ---------------- Main getitem --------------------
    # ==================================================

    def __getitem__(self, idx):
        uid, img_path, class_target, center = self.samples[idx]

        img = np.transpose(load_volume(img_path), (2, 1, 0))
        seg_path = find_existing_file(
            self.labels_path, uid, label_candidates
        )
        seg = np.transpose(load_volume(seg_path), (2, 1, 0))

        img_p = self._extract_patch_at_center(img, center)
        seg_p = self._extract_patch_at_center(seg, center)

        img_t = torch.from_numpy(img_p).unsqueeze(0).float()
        seg_t = torch.from_numpy(seg_p).unsqueeze(0).float()

        if self.transforms:
            img_t = self.transforms(img_t)

        if self.use_labels:
            img_t = torch.cat([img_t, seg_t], dim=0)

        return {
            "uid": uid,
            "source": img_t,
            "segmentation": seg_t,
            "target": torch.tensor(class_target, dtype=torch.long),
            "patch_center": tuple(center.tolist()),
        }
    def __len__(self):
        return len(self.samples)

"""""
    def _get_sampling_coords(self, uid, seg):
        if uid in self._coord_cache:
            return self._coord_cache[uid]

        fg = np.argwhere(seg > 0)

        labeled, n = ndimage.label(seg > 0)
        if n > 0:
            roi_centers = np.array(
                ndimage.center_of_mass(seg > 0, labeled, range(1, n + 1))
            ).astype(int)
        else:
            roi_centers = np.empty((0, 3), dtype=int)

        margin = self.patch_size // 2
        dilated = ndimage.binary_dilation(seg > 0, iterations=int(margin.max()))
        bg = np.argwhere(dilated == 0)

        if len(bg) == 0:
            bg = np.argwhere(seg == 0)

        self._coord_cache[uid] = (fg, roi_centers, bg)
        return fg, roi_centers, bg

    # ------------------------------------------------------------------

    def _extract_patch(self, vol, center):
        ps = self.patch_size
        start = center - ps // 2
        end = start + ps

        patch = np.zeros(ps, dtype=vol.dtype)

        s0 = np.clip(start, 0, vol.shape)
        e0 = np.clip(end, 0, vol.shape)

        dst_s = np.maximum(0, -start)
        dst_e = dst_s + (e0 - s0)

        patch[
            dst_s[0]:dst_e[0],
            dst_s[1]:dst_e[1],
            dst_s[2]:dst_e[2],
        ] = vol[
            s0[0]:e0[0],
            s0[1]:e0[1],
            s0[2]:e0[2],
        ]

        return patch

    # ------------------------------------------------------------------
    # Main getitem
    # ------------------------------------------------------------------

    def __getitem__(self, idx):

        # -------------------------------------------------
        # Select subject
        # -------------------------------------------------
        if self.return_full_image:
            uid, img_path, target = self.images[idx]
        else:
            uid, img_path, target = random.choice(self.images)
        #print("Selected UID:", uid)
        # -------------------------------------------------
        # Load image and label
        # -------------------------------------------------
        img = load_volume(img_path)
        seg_path = find_existing_file(self.labels_path, uid, label_candidates)
        seg = load_volume(seg_path)
        #print("img and seg loaded")
        img = np.transpose(img, (2, 1, 0))  # (D, H, W)
        seg = np.transpose(seg, (2, 1, 0))  # (D, H, W)

        fg, roi, bg = self._get_sampling_coords(uid, seg)
        #print(f"Foreground voxels: {len(fg)}, Background voxels: {len(bg)}, ROIs: {len(roi)}")
        want_fg = self.rng.random() < self.p_foreground

        if want_fg and len(fg) > 0:
            if len(roi) > 0 and self.rng.random() < 0.5:
                center = roi[self.rng.integers(len(roi))]
            else:
                center = fg[self.rng.integers(len(fg))]
        
            class_target = target
        else:
            if len(bg) == 0:
                center = fg[self.rng.integers(len(fg))]
            else:
                center = bg[self.rng.integers(len(bg))]
            class_target = self.background_class

        # -------------------------------------------------
        # Extract patches
        # -------------------------------------------------
        img_p = self._extract_patch(img, center)
        seg_p = self._extract_patch(seg, center)
        #print("patch extracted")
        img_t = torch.from_numpy(img_p).unsqueeze(0).float()
        seg_t = torch.from_numpy(seg_p).unsqueeze(0).float()

        if self.transforms:
            img_t = self.transforms(img_t)

        # Concatenate label as channel (same as basedataset)
        if self.use_labels:
            img_t = torch.cat([img_t, seg_t], dim=0)

        return {
            "uid": uid,
            "source": img_t,
            "segmentation": seg_t,
            "target": torch.tensor(class_target, dtype=torch.long),
            "patch_center": tuple(center.tolist()),
        }
    """""


from monai.transforms import RandCropByPosNegLabeld


class ColonCancer_monai(basedataset):
    """
    Colon Cancer dataset with MONAI foreground/background patch sampling.
    Supports:
      - full image or patch sampling
      - 1 or 2 image channels (image [+ label])
      - always returns label separately (for debugging)
    """

    def __init__(
        self,
        patch_size,
        num_patches_per_epoch,
        p_foreground=0.7,
        background_class=0,
        transforms=None,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            transforms=transforms,
            num_patches_per_epoch=num_patches_per_epoch,
            **kwargs,
        )

        self.patch_size = tuple(patch_size)
        self.p_foreground = p_foreground
        self.background_class = background_class

        self.df = pd.read_csv(self.labels_file)

        # MONAI pos/neg sampler
        pos = int(self.p_foreground * 10)
        neg = max(1, 10 - pos)

        self.monai_sampler = RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=self.patch_size,
            pos=pos,
            neg=neg,
            num_samples=1,
            allow_smaller=True,
        )

    def __getitem__(self, idx):
        # -------------------------------------------------
        # Select subject (random, like your base dataset)
        # -------------------------------------------------
        if self.return_full_image:
            uid, img_path, class_target_img = self.images[idx]
        else:
            uid, img_path, class_target_img = random.choice(self.images)

        # -------------------------------------------------
        # Load image and label
        # -------------------------------------------------
        img = load_volume(img_path)
        seg_path = find_existing_file(self.labels_path, uid, label_candidates)
        seg = load_volume(seg_path) 

        img = np.transpose(img, (2, 1, 0))  # (D,H,W)
        seg = np.transpose(seg, (2, 1, 0))  # (D,H,W)

        img_t = torch.from_numpy(img).unsqueeze(0)   # (1,D,H,W)
        seg_t = torch.from_numpy(seg).unsqueeze(0)   # (1,D,H,W)

        # -------------------------------------------------
        # FULL IMAGE MODE (no sampling)
        # -------------------------------------------------
        if self.return_full_image:
            image_out = img_t

            if self.use_labels:
                image_out = torch.cat([image_out, seg_t], dim=0)

            sample = {
                "uid": uid,
                "source": image_out.float(),
                "segmentation": seg_t.float(),
                "target": torch.tensor(class_target_img, dtype=torch.long),
            }

            if self.transforms:
                sample = self.transforms(sample)

            return sample

        # -------------------------------------------------
        # PATCH SAMPLING MODE (MONAI)
        # -------------------------------------------------
        
        data = {
            "image": img_t,
            "label": seg_t,
        }

        cropped = self.monai_sampler(data)[0]

        image_patch = cropped["image"]   # (1,ps,ps,ps)
        label_patch = cropped["label"]   # (1,ps,ps,ps)
        if image_patch.shape[1:] != self.patch_size:
                image_patch= pad_to_shape(image_patch, self.patch_size)
        if label_patch.shape[1:] != self.patch_size:
                label_patch = pad_to_shape(label_patch, self.patch_size)

        # patch-level foreground check
        has_fg = torch.any(label_patch > 0)
        class_target = (
            class_target_img if has_fg else self.background_class
        )

        # optionally concatenate label as 2nd channel
        image_out = image_patch
        if self.use_labels:
            image_out = torch.cat([image_out, label_patch], dim=0)

        sample = {
            "uid": uid,
            "source": image_out.float(),
            "segmentation": label_patch.float(),   # always returned
            "target": torch.tensor(class_target, dtype=torch.long),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return self.num_patches_per_epoch
