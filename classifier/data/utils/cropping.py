import json
import os
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from classifier.data.utils.functions_utils import load_volume


def voxels_from_mm(spacing, margin_mm: Tuple[float, float, float]):
    """Convert physical margin in mm to voxel units (rounding up)."""
    return tuple([int(np.ceil(mm / sp)) for mm, sp in zip(margin_mm, spacing)])


def get_bbox_from_mask_with_margin(
    mask: np.ndarray, margin_vox: Tuple[int, int, int]
) -> List[Tuple[int, int]]:
    """Compute bounding box from nonzero mask with added voxel margin."""
    nonzero = np.where(mask > 0)
    if len(nonzero[0]) == 0:
        # no label found
        return [(0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2])]

    bbox = []
    for d in range(3):
        start = max(int(np.min(nonzero[d])) - margin_vox[d], 0)
        end = min(int(np.max(nonzero[d])) + margin_vox[d] + 1, mask.shape[d])
        bbox.append((start, end))
    return bbox


def crop_to_bbox_no_channels(image, bbox: Sequence[Sequence[int]]):
    """Crops 3D image to bounding box (no channels)."""
    resizer = tuple(slice(start, end) for start, end in bbox)
    return image[resizer]


def crop_to_bbox(data: np.ndarray, bbox: Sequence[Sequence[int]]):
    """Crops 3D/4D array per channel to given bounding box."""
    cropped_data = [
        crop_to_bbox_no_channels(data[c], bbox) for c in range(data.shape[0])
    ]
    return np.stack(cropped_data)


def crop_to_label_region(
    data: np.ndarray,
    crop_mask: np.ndarray,
    seg: np.ndarray,
    spacing: Tuple[float, float, float],
    margin_min: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Crop image & segmentation to a region of interest defined by the label (segmentation mask).

    Args:
        data (np.ndarray): 4D image [C, D, H, W]
        seg (np.ndarray): 3D segmentation [D, H, W]
        spacing (Tuple[float]): voxel spacing in mm
        margin_min (float): margin (in mm) added around the labeled region

    Returns:
        cropped_data (np.ndarray): cropped image
        cropped_seg (np.ndarray): cropped segmentation
        bbox (List[Tuple[int, int]]): voxel bounding box
    """
    margin_vox = voxels_from_mm(spacing, (margin_min, margin_min, margin_min))
    bbox = get_bbox_from_mask_with_margin(crop_mask, margin_vox)

    if data.ndim == 4:
        data_cropped = crop_to_bbox(data, bbox)
    elif data.ndim == 3:
        data_cropped = crop_to_bbox_no_channels(data, bbox)

    seg_cropped = crop_to_bbox_no_channels(seg, bbox)

    return data_cropped, seg_cropped, bbox


def batch_crop_and_save(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    margin_min: float = 15.0,
    crop_labels: bool = True,
    split: str = "Tr",
    crop_to_colon: bool = False,
):
    """
    Crop all images in a dataset to their label-defined ROI and save results + metadata.

    Args:
        images_dir (str): directory with input NIfTI images (*.nii or *.nii.gz)
        labels_dir (str): directory with corresponding label maps
        output_dir (str): output folder to store cropped results
        margin_min (float): margin (in mm) around labeled region
        crop_labels (bool): whether to crop the label maps along with images
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    cropped_images_dir = output_dir / "images_cropped"
    cropped_labels_dir = output_dir / "labels_cropped"
    cropped_images_dir.mkdir(parents=True, exist_ok=True)
    if crop_labels:
        cropped_labels_dir.mkdir(parents=True, exist_ok=True)

    records = []
    original_shapes = []
    cropped_shapes = []

    image_files = sorted(images_dir.glob("*.nii*"))
    print(f"Found {len(image_files)} images to process from {images_dir}")

    for img_path in tqdm(image_files, desc="Cropping dataset"):
        uid = int(img_path.stem.replace("_0000.nii", ""))
        out_img_path = cropped_images_dir / f"{uid}.nii.gz"
        out_label_path = cropped_labels_dir / f"{uid}.nii.gz" if crop_labels else None

        if out_img_path.exists():
            print(f"Skipping {uid}, already cropped.")
            continue

        label_path = labels_dir / f"{uid}.nii.gz"
        colon_label_path = Path(os.environ.get("auto_seg_path")) / Path(f"{uid}.nii.gz")

        if not label_path.exists():
            print(f"Label not found for {uid}, skipping.")
            continue

        img_nii = nib.load(str(img_path))
        seg_nii = nib.load(str(label_path))
        data = img_nii.get_fdata().astype(np.float32)
        seg = seg_nii.get_fdata().astype(np.uint8)

        ######################################################################################
        # seg_before = seg.sum()
        # seg = remove_all_but_largest_component(seg)
        # seg = seg.astype(np.uint8)
        # print(f"[{uid}] Kept {seg.sum()} / {seg_before} voxels after cleaning.")
        ########################################################################################
        if (seg.sum() == 0 or crop_to_colon) and colon_label_path.exists():
            colon_seg_nii = nib.load(str(colon_label_path))
            colon_seg = colon_seg_nii.get_fdata().astype(np.uint8)
            crop_mask = colon_seg
        else:
            crop_mask = seg
        if data.ndim == 3:
            data = data[None, ...]

        spacing = img_nii.header.get_zooms()[:3]
        original_shape = list(data.shape[1:])  # exclude channel dim
        original_shapes.append(original_shape)

        # ---- Crop to ROI
        data_cropped, seg_cropped, bbox = crop_to_label_region(
            data=data,
            crop_mask=crop_mask,
            seg=seg,
            spacing=spacing,
            margin_min=margin_min,
        )
        cropped_shape = list(data_cropped.shape[1:])
        cropped_shapes.append(cropped_shape)

        # ---- Save cropped image
        cropped_img_nii = nib.Nifti1Image(data_cropped[0], affine=img_nii.affine)
        nib.save(cropped_img_nii, str(out_img_path))

        if crop_labels:
            cropped_seg_nii = nib.Nifti1Image(seg_cropped, affine=seg_nii.affine)
            nib.save(cropped_seg_nii, str(out_label_path))

        # ---- Record metadata
        record = {
            "UID": uid,
            "img_path": str(out_img_path),
            "bbox": bbox,
            "original_shape": original_shape,
            "cropped_shape": cropped_shape,
        }
        if crop_labels:
            record["label_path"] = str(out_label_path)
        records.append(record)

    # ---- Save per-case metadata
    df = pd.DataFrame(records)
    df.to_csv(output_dir.parent / f"cropping_metadata_{split}.csv", index=False)

    # ---- Compute global shape statistics
    def compute_shape_stats(shapes):
        shapes = np.array(shapes)
        total_voxels = np.prod(shapes, axis=1)
        return {
            "mean_shape": shapes.mean(axis=0).tolist(),
            "median_shape": np.median(shapes, axis=0).tolist(),
            "min_shape": shapes.min(axis=0).tolist(),
            "max_shape": shapes.max(axis=0).tolist(),
            "10pct_shape": np.percentile(shapes, 10, axis=0).tolist(),
            "90pct_shape": np.percentile(shapes, 90, axis=0).tolist(),
            "mean_total_voxels": float(total_voxels.mean()),
            "median_total_voxels": float(np.median(total_voxels)),
            "min_total_voxels": float(total_voxels.min()),
            "max_total_voxels": float(total_voxels.max()),
        }

    stats = {
        "original_shapes": compute_shape_stats(original_shapes),
        "cropped_shapes": compute_shape_stats(cropped_shapes),
    }

    # ---- Save stats as JSON
    with open(output_dir.parent / f"cropping_shape_statistics_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n Cropped {len(df)} cases.")
    print(f" Saved metadata CSV → {output_dir / 'cropping_metadata.csv'}")
    print(f" Saved shape statistics JSON → {output_dir / 'shape_statistics.json'}")


def crop_nnunet_data(
    input_dir: str, output_dir: str, margin_min: float = 20.0, overwrite: bool = True
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only get image files (exclude _seg.b2nd)
    image_files = sorted([f for f in input_dir.glob("*.b2nd") if "_seg" not in f.name])
    print(f"Found {len(image_files)} images to process from {input_dir}")

    for img_path in tqdm(image_files, desc="Cropping dataset"):
        print(img_path)
        uid = int(img_path.stem)

        out_img_path = output_dir / f"{uid}.npz"
        out_label_path = output_dir / f"{uid}_seg.npz"

        if not overwrite and out_img_path.exists():
            print(f"Skipping {uid}, already cropped.")
            continue

        label_path = input_dir / f"{uid}_seg.b2nd"
        if not label_path.exists():
            print(f"Label not found for {uid}, skipping.")
            continue

        metadata_path = input_dir / f"{uid}.pkl"
        if not metadata_path.exists():
            print(f"Metadata not found for {uid}, skipping.")
            continue

        # ---- Load metadata
        with open(metadata_path, "rb") as f:
            prop = pickle.load(f)

        spacing = prop.get("target_spacing", None)
        ##################################################################################################################################
        # spacing = [1, 1, 1]
        ###################################################################################################################################
        # ---- Load .b2nd image and segmentation
        data = load_volume(img_path)
        seg = load_volume(label_path)

        if data.ndim == 3:
            data = data[None, ...]

        # ---- Crop to ROI

        data_cropped, seg_cropped, bbox = crop_to_label_region(
            data, seg, spacing, margin_min=margin_min
        )

        data_cropped = np.transpose(data_cropped[0], (2, 1, 0))
        seg_cropped = np.transpose(seg_cropped, (2, 1, 0))

        # ---- Save cropped image and segmentation
        np.savez_compressed(out_img_path, image=data_cropped)
        np.savez_compressed(out_label_path, image=seg_cropped)
