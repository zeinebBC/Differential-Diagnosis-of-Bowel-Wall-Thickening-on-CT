"""
###################################################### run before using refined labels ###########################################################################################

import os
import shutil
from glob import glob
import SimpleITK as sitk
import numpy as np
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component


# ===========================================================
#   SLICE UTILITIES
# ===========================================================

def find_annotated_slices(label, axis):

    if axis == "axial":
        reduce_axes = (1, 2)
    elif axis == "coronal":
        reduce_axes = (0, 2)
    elif axis == "sagittal":
        reduce_axes = (0, 1)
    else:
        raise ValueError("Invalid axis")

    return np.where(np.any(label > 0, axis=reduce_axes))[0]


# ===========================================================
#   GAP-AWARE INTERPOLATION
# ===========================================================

def interpolate_gap(lbl, k0, k1):

    for k in range(k0 + 1, k1):
        w = (k - k0) / float(k1 - k0)
        lbl[k] = ((1 - w) * lbl[k0] + w * lbl[k1]) > 0
    return lbl


def interpolate_missing_slices(label, axis="axial", max_gap=5):

    lbl = label.copy()

    # Reorient so slice dimension = 0
    if axis == "coronal":
        lbl = np.transpose(lbl, (1, 0, 2))
    elif axis == "sagittal":
        lbl = np.transpose(lbl, (2, 1, 0))

    annotated = find_annotated_slices(lbl, "axial")

    if len(annotated) < 2:
        return label

    for i in range(len(annotated) - 1):
        k0, k1 = annotated[i], annotated[i + 1]
        gap = k1 - k0 - 1

        if 1 <= gap <= max_gap:
            lbl = interpolate_gap(lbl, k0, k1)

    # Restore orientation
    if axis == "coronal":
        lbl = np.transpose(lbl, (1, 0, 2))
    elif axis == "sagittal":
        lbl = np.transpose(lbl, (2, 1, 0))

    return lbl.astype(label.dtype)


# ===========================================================
#   RESAMPLING
# ===========================================================

def resample_label_to_image(label_path, image_ref_path):
    label_sitk = sitk.ReadImage(str(label_path))
    ref_sitk = sitk.ReadImage(str(image_ref_path))

    return sitk.Resample(
        label_sitk,
        ref_sitk,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_sitk.GetPixelID(),
    )


# ===========================================================
#   PATHS
# ===========================================================

label_dir = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr/labels/final"
image_dir = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr"

out_image_dir = "/data/colon_cancer/CC_Detection/raw_data/Dataset108_CC/imagesTr"
out_label_dir = "/data/colon_cancer/CC_Detection/raw_data/Dataset108_CC/labelsTr"

os.makedirs(out_image_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

label_files = sorted(glob(os.path.join(label_dir, "*.nii.gz")))
print(f"Found {len(label_files)} label files")


# ===========================================================
#   MAIN PIPELINE
# ===========================================================

MAX_GAP = 4

for label_path in label_files:
    label_name = os.path.basename(label_path)
    uid = label_name.split("_")[0]
    img_path = os.path.join(image_dir, label_name)

    if not os.path.exists(img_path):
        print(f"WARNING: No matching image for {label_name}")
        continue

    # Copy image
    new_img_name = f"{uid}_0000.nii.gz"
    shutil.copy2(img_path, os.path.join(out_image_dir, new_img_name))

    # Load + resample label
    label_resampled = resample_label_to_image(label_path, img_path)
    label_np = sitk.GetArrayFromImage(label_resampled).astype(np.uint8)
    label_np[label_np == 2] = 0

    # --- Report gaps BEFORE interpolation ---
    annotated = find_annotated_slices(label_np, "axial")
    print(f"{uid}: annotated slices = {annotated}")

    for i in range(len(annotated) - 1):
        gap = annotated[i + 1] - annotated[i] - 1
        if gap > 0:
            print(f"  gap of {gap} slices between {annotated[i]} and {annotated[i+1]}")

    # --- Interpolate small gaps FIRST ---
    mask_interp = interpolate_missing_slices(
        label_np,
        axis="axial",
        max_gap=MAX_GAP,
    )

    # --- NOW keep only largest component ---
    mask_final = remove_all_but_largest_component(mask_interp).astype(np.uint8)

    if not np.array_equal(mask_final, label_np):
        print(f"✓ Updated mask for {uid}")
    else:
        print(f"Keeping original mask for {uid}")

    # Save label
    final_label_sitk = sitk.GetImageFromArray(mask_final)
    final_label_sitk.CopyInformation(label_resampled)

    new_label_name = f"{uid}.nii.gz"
    sitk.WriteImage(final_label_sitk, os.path.join(out_label_dir, new_label_name))

    print(f"Processed {uid}: Image={new_img_name}, Label={new_label_name}")

print("Done!")
"""

################################################################ compute full metrics using seg-metrics  ##############################################

################################################################ compute full metrics using seg-metrics  ##############################################

#!/usr/bin/env python3
import json
import os

import nibabel as nib
import numpy as np
import seg_metrics.seg_metrics as sg

# -----------------------------
# Configuration
# -----------------------------
pred_dir = "/data/colon_cancer/CC_Detection/raw_data/Decathlon/predictionsDecathlon"
gt_dir = "/data/colon_cancer/CC_Detection/raw_data/Decathlon/labelsTs"

output_file = "metrics_results_Ts_Decathlon.json"

# Thresholds to categorize dice scores
bad_thresh = 0.4
medium_thresh = 0.7

# seg-metrics metrics (set the metrics you want to)
METRICS = [
    "dice",
    "precision",
    "recall",
    "fpr",
    "fnr",
    # "msd",
    # "hd95",
]


# -----------------------------
# Helper functions
# -----------------------------
def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata().astype(np.uint8), img.header


def get_spacing(gt_path, pred_path):
    """
    Returns spacing in (Z, Y, X) order.
    Prefers GT spacing.
    """
    if os.path.exists(gt_path):
        hdr = nib.load(gt_path).header
    else:
        hdr = nib.load(pred_path).header
    return hdr.get_zooms()[:3]


def categorize_dice(dice_val):
    if dice_val < bad_thresh:
        return "bad"
    elif dice_val < medium_thresh:
        return "medium"
    else:
        return "good"


def compute_metrics(pred, gt, spacing):
    """
    Wrapper around seg-metrics
    """
    metrics = sg.write_metrics(
        labels=[1],
        pred_img=pred.astype(np.uint8),
        gdth_img=gt.astype(np.uint8),
        metrics=METRICS,
        spacing=spacing,
    )
    return {m: float(metrics[0][m][0]) for m in METRICS}


# -----------------------------
# Main
# -----------------------------
def main():
    results = {}
    categories = {"bad": [], "medium": [], "good": []}

    all_metrics = {m: [] for m in METRICS}

    pred_files = sorted(
        f for f in os.listdir(pred_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
    )

    for pf in pred_files:
        print(f"\nProcessing {pf}...")
        pred_path = os.path.join(pred_dir, pf)
        gt_path = os.path.join(gt_dir, pf)

        if not os.path.exists(gt_path):
            print(f"⚠️ GT not found for {pf}, skipping")
            continue

        pred, _ = load_nifti(pred_path)
        gt, _ = load_nifti(gt_path)

        spacing = get_spacing(gt_path, pred_path)

        metrics = compute_metrics(pred, gt, spacing)
        dice_val = metrics["dice"]

        category = categorize_dice(dice_val)
        categories[category].append(pf)

        results[pf] = {
            "metrics": metrics,
            "category": category,
        }

        for m in METRICS:
            all_metrics[m].append(metrics[m])

        print(
            f"{pf}: Dice={dice_val:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f} -> {category}"
        )

    # -----------------------------
    # Aggregation
    # -----------------------------
    aggregated = {
        "mean": {m: float(np.mean(all_metrics[m])) for m in METRICS},
        "median": {m: float(np.median(all_metrics[m])) for m in METRICS},
        "min": {m: float(np.min(all_metrics[m])) for m in METRICS},
        "max": {m: float(np.max(all_metrics[m])) for m in METRICS},
        "counts": {k: len(v) for k, v in categories.items()},
    }

    output = {
        "per_file": results,
        "aggregated": aggregated,
        "categories": categories,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ Done! Results saved to:", output_file)
    print(json.dumps(aggregated, indent=2))


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
