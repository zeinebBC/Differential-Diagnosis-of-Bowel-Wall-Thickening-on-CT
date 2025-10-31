import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from typing import List
import re

def list_nifti_files(input_dir: str) -> List[Path]:
    paths: List[Path] = []
    for ext in (".nii", ".nii.gz"):
        paths.extend(Path(input_dir).rglob(f"*{ext}"))
    paths = sorted(list(set(paths)))
    return paths

def get_number_from_filename(str: str) -> str:
    """
    Extract the leading number from filenames like '123_0000.nii.gz'.
    """
    match = re.match(r"(\d+)_", str)
    if match:
        return match.group(1)
    raise ValueError(f"No leading number found in filename {str}")

def run_case(input_path: Path, output_root: Path, overwrite: bool, task:str, roi_subset:str=None ) -> None:
    """
    Run TotalSegmentator using the Python API for a single case.
    Saves the output as colon_mask.nii.gz in the case folder.
    """
    output_path = output_root / task
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    case_id = get_number_from_filename(input_path.stem)
    output_path = Path(output_path) /f"{case_id}.nii.gz"

    
    if output_path.exists() and not overwrite:
        return

    input_img = nib.load(input_path)
    if roi_subset:
        output_img = totalsegmentator(input=input_img,task=task, roi_subset=roi_subset)
    else: 
        output_img = totalsegmentator(input=input_img,task=task)
    nib.save(output_img,output_path)


   


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch TotalSegmentator colon inference (Python API)")
    parser.add_argument("--input", required=True, help="Folder with input NIfTI volumes")
    parser.add_argument("--output", required=True, help="Output folder for masks")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--task",type=str, default="total", help="Name of the segmentation task" )
    parser.add_argument("--roi_subset",type=str, nargs='+', default=None, help="specify the region(s) of interest")

    
    args = parser.parse_args()

    images = list_nifti_files(args.input)
    if not images:
        raise SystemExit(f"No NIfTI files found in {args.input}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(run_case, img, Path(args.output), args.overwrite, args.task, args.roi_subset) for img in images
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Segmentation failed: {e}")


if __name__ == "__main__":
    main()



#micromamba create -n ttseg_env python=3.12.3
#conda install pytorch==2.4.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge
#pip install TotalSegmentator
#conda activate ttseg_env

#python -m scripts.run_totalseg --input /data/colon_cancer/Task100_Colon/raw_splitted/imagesTr/ --output /data/benchaaben/colon_seg/outputs