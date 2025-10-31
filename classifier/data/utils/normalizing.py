import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from glob import glob
from tqdm import tqdm 
import shutil 

def window_and_normalize(image: np.ndarray, window_min=-100, window_max=500):
    """
    Apply CT windowing and normalize to [0, 1].
    """
    image = np.clip(image, window_min, window_max)
    image = (image - window_min) / (window_max - window_min)
    return image.astype(np.float32)


def process_and_window_dataset(
    images_dir: str,
    output_dir: str,
    window_min: float = -100,
    window_max: float = 500,
    overwrite: bool = False,
    split:str ="Tr",

):
    """
    Apply windowing and normalization to a folder of CT NIfTI images,
    save them as .npz files, and compute dataset-level intensity statistics.

    Args:
        images_dir (str): Folder containing input .nii/.nii.gz images.
        output_dir (str): Folder to save processed .npz files.
        window_min (float): Minimum Hounsfield value for windowing.
        window_max (float): Maximum Hounsfield value for windowing.
        overwrite (bool): Whether to overwrite existing files.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(glob(str(images_dir / "*.nii*")))
    print(f"Found {len(image_paths)} images to process in {images_dir}")

    # Store per-image statistics
    stats_records = []

    for img_path in tqdm(image_paths, desc="Windowing + Normalizing"):
        uid = int(Path(img_path).stem.replace(".nii", ""))
        save_path = output_dir / f"{uid}.npz"

        if not overwrite and save_path.exists():
            print(f"Skipping {uid} (already processed)")
            continue

        # Load image
        img_nii = nib.load(str(img_path))
        img_np = img_nii.get_fdata().astype(np.float32)

        # Apply windowing + normalization
        img_proc = window_and_normalize(img_np, window_min, window_max)
        
        # Save compressed image
        np.savez_compressed(save_path, image=img_proc)

        # Compute per-image intensity stats
        img_stats = {
            "UID": uid,
            "mean": float(np.mean(img_proc)),
            "std": float(np.std(img_proc)),
            "min": float(np.min(img_proc)),
            "max": float(np.max(img_proc)),
        }
        stats_records.append(img_stats)

    # --- Dataset-level statistics
    df = pd.DataFrame(stats_records)
    dataset_stats = {
        "mean_of_means": df["mean"].mean(),
        "mean_of_stds": df["std"].mean(),
        "global_min": df["min"].min(),
        "global_max": df["max"].max(),
    }

    # Save per-image + dataset-level statistics
    stats_csv_path = output_dir.parent / f"intensity_statistics_{split}.csv"
    df.to_csv(stats_csv_path, index=False)

    print(f"\nSaved windowed dataset to {output_dir}")
    print(f"Saved intensity statistics CSV to {stats_csv_path}")
    print(f"Global stats: {dataset_stats}")

    
    print(f"Removing temporary resampling folder")
    shutil.rmtree(images_dir, ignore_errors=True)
    return dataset_stats
