
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
import shutil 

def resample_image(image, target_spacing, is_label=False, interpolator='cubic'):
    """Resample a SimpleITK image to the target spacing."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    
    new_size = [
        int(round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        if interpolator == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolator == 'cubic':
            resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)


def batch_resample_and_save(
    root_dir: str,
    output_dir: str,
    target_spacing: tuple = (0.7, 0.7, 0.8),
    overwrite: bool = False,
    split:str ="Tr",

):
    """
    Batch resample images (and optionally labels) to a target spacing and save results + metadata.
    
    Args:
        images_dir (str): directory containing input NIfTI images (*.nii or *.nii.gz)
        output_dir (str): base directory to save resampled outputs
        target_spacing (tuple): desired voxel spacing (z, y, x)
        labels_dir (str, optional): directory with label files (if provided)
        overwrite (bool): whether to overwrite existing files
    """
    images_dir = Path(root_dir) / "images_cropped"
    labels_dir = Path(root_dir) / "labels_cropped" 
    resampled_images_dir = Path(output_dir) / "images_resampled"
    
    resampled_images_dir.mkdir(parents=True, exist_ok=True)
    if  labels_dir.exists():
        resampled_labels_dir = Path(output_dir) / "labels_resampled"
        resampled_labels_dir.mkdir(parents=True, exist_ok=True)

    records = []
    shapes_orig = []
    shapes_resampled = []
    spacings_orig = []

    image_files = sorted(list(images_dir.glob("*.nii")) + list(images_dir.glob("*.nii.gz")))
    print(f"Found {len(image_files)} images in {images_dir}")

    for img_path in tqdm(image_files, desc="Resampling dataset"):
        uid = int(img_path.stem.replace("_0000.nii", "").replace(".nii", ""))
        out_img_path = resampled_images_dir / f"{uid}.nii.gz"
       

        if not overwrite and out_img_path.exists():
            print(f"Skipping {uid}, already resampled.")
            continue

        # Load image
        image = sitk.ReadImage(str(img_path))
        
        orig_spacing = image.GetSpacing()
        orig_size = image.GetSize()

        print(f"Processing {uid}: original size {orig_size}, spacing {orig_spacing}")
        shapes_orig.append(orig_size)
        spacings_orig.append(orig_spacing)
        # --- Resample image
        res_image = resample_image(image, target_spacing=target_spacing)
        
        shapes_resampled.append(res_image.GetSize())

        # --- Save resampled image
        sitk.WriteImage(res_image, str(out_img_path))

        # --- Optional label resampling
        if labels_dir.exists():
            label_path = Path(labels_dir) / f"{uid}.nii.gz"
            out_label_path = resampled_labels_dir / f"{uid}_seg.nii.gz"
            if label_path.exists():
                label = sitk.ReadImage(str(label_path))
                res_label = resample_image(label, target_spacing=target_spacing, is_label=True)
                sitk.WriteImage(res_label, str(out_label_path))
            else:
                print(f"Warning: label not found for {uid}, skipping label resampling.")

        # Record metadata
        records.append({
            "UID": uid,
            "orig_size": orig_size,
            "orig_spacing": orig_spacing,
            "resampled_size": res_image.GetSize(),
            "resampled_spacing": target_spacing,
            "img_path": str(out_img_path),
            "label_path": str(out_label_path) if labels_dir else None
        })

    # --- Save metadata per image
    df = pd.DataFrame(records)
    csv_path = output_dir.parent / f"resampling_metadata_{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata for {len(df)} cases → {csv_path}")

    # --- Compute dataset-level shape statistics
    def compute_shape_stats(shapes_arr):
        total_voxels = np.prod(shapes_arr, axis=1)
        stats = {
            "mean_shape": shapes_arr.mean(axis=0).tolist(),
            "median_shape": np.median(shapes_arr, axis=0).tolist(),
            "min_shape": shapes_arr.min(axis=0).tolist(),
            "max_shape": shapes_arr.max(axis=0).tolist(),
            "10pct_shape": np.percentile(shapes_arr, 10, axis=0).tolist(),
            "90pct_shape": np.percentile(shapes_arr, 90, axis=0).tolist(),
            "mean_total_voxels": float(total_voxels.mean()),
            "median_total_voxels": float(np.median(total_voxels)),
            "min_total_voxels": float(total_voxels.min()),
            "max_total_voxels": float(total_voxels.max())
        }
        return stats

    shapes_orig_arr = np.array(shapes_orig)
    shapes_resampled_arr = np.array(shapes_resampled)
    spacings_orig_arr = np.array(spacings_orig)

    stats = {
        "original_shapes": compute_shape_stats(shapes_orig_arr),
        "resampled_shapes": compute_shape_stats(shapes_resampled_arr),
        "original_spacings": compute_shape_stats(spacings_orig_arr)
    }

    stats_path = output_dir.parent / f"shape_statistics_{split}.json"
    pd.DataFrame(stats).to_json(stats_path, indent=4)
    print(f"\nSaved dataset-level shape statistics → {stats_path}")

 
    print(f" Removing temporary cropping folders:")
    shutil.rmtree(images_dir.parent, ignore_errors=True)
    

    return stats
