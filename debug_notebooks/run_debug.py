"""
import os
import shutil

# =========================
# CONFIG
# =========================
folder_a = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr"      # source folder
folder_b = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr/labels/original"      # reference folder
out_folder = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr_extracted" # destination

os.makedirs(out_folder, exist_ok=True)

# =========================
# COLLECT FILENAMES
# =========================
files_a = set(os.listdir(folder_a))
files_b = set(os.listdir(folder_b))

# =========================
# FIND & MOVE FILES
# =========================
to_move = files_a - files_b

print(f"Found {len(to_move)} files to move")

for fname in sorted(to_move):
    src = os.path.join(folder_a, fname)
    dst = os.path.join(out_folder, fname)

    if os.path.isfile(src):
        shutil.move(src, dst)
        print(f"Moved: {fname}")
    else:
        print(f"Skipped (not a file): {fname}")

print("Done.")
"""

import os
import shutil

src_dir = "/data/colon_cancer/CC_Detection/raw_data/Dataset108_CC/imagesTs"
dst_dir = "/data/colon_cancer/Task101_Colon/raw_splitted/imagesTr"


os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith(".nii.gz"):
        shutil.copy2(
            os.path.join(src_dir, fname),
            os.path.join(dst_dir, fname)
        )
        print(f"Copied: {fname}")
