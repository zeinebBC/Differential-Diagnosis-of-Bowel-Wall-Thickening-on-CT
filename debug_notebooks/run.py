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
    """Return sorted indices of slices containing label > 0."""
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
    """Interpolate slices strictly between k0 and k1."""
    for k in range(k0 + 1, k1):
        w = (k - k0) / float(k1 - k0)
        lbl[k] = ((1 - w) * lbl[k0] + w * lbl[k1]) > 0
    return lbl


def interpolate_missing_slices(label, axis="axial", max_gap=5):
    """
    Interpolate ONLY small gaps (<= max_gap) between annotated slices.
    """
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
