
import json
import os
from pathlib import Path
import pandas as pd


def generate_target_mapping_method(labels_file, path_root, dataset_name):
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
        pd.read_csv(labels_file) if labels_file.exists() else pd.DataFrame()
    )
    existing_uids = (
        set(existing_df["UID"].astype(str)) if not existing_df.empty else set()
    )

    mapping = []
    num_div, num_cancer = 0, 0

    # Loop over both train and test folders
    for split in ["Tr", "Ts"]:
        images_folder = path_root / f"raw_data/{dataset_name}/images{split}"
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
    final_df.to_csv(labels_file, index=False)

    print(f"Saved classification labels for {len(final_df)} images")
    print(f"Diverticulitis: {num_div}, Colon cancer: {num_cancer}")