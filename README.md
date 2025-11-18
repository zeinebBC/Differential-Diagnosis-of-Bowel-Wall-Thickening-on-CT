### README

This repository provides a complete deep learning pipeline for **colon cancer detection** using CT scans.  
The system consists of two main stages:

1. **Segmentation** — using `nnU-Net v2` to localize bowel wall thickening.  
2. **Classification** — using a custom `ResNet` model to distinguish between **cancer** and **diverticulitis**.

### Installation
- Create and activate a conda environment using the provided requirements file:
```bash
conda create -n cc_detection --file requirements.txt
conda activate cc_detection
```

### Dataset Structure
Your dataset directory must follow this structure. Please respect the filenames and formats.
```
root
├── pp_data
│   └── Dataset100_CC
├── raw_data
│   └── Dataset100_CC
│       ├── dataset.json
│       ├── imagesTr
│       │   ├── 1_0000.nii.gz
│       │   ├── 2_0000.nii.gz
│       │   ├── 3_0000.nii.gz
│       │   └── ...
│       ├── imagesTs
│       │   ├── 4_0000.nii.gz
│       │   ├── 5_0000.nii.gz
│       │   ├── 6_0000.nii.gz
│       │   └── ...
│       ├── labels.csv
│       ├── labelsTr
│       │   ├── 1.nii.gz
│       │   ├── 2.nii.gz
│       │   ├── 3.nii.gz
│       │   └── ...
│       └── labelsTs (optional)
│           ├── 4.nii.gz
│           ├── 5.nii.gz
│           ├── 6.nii.gz
│           └── ...
```

- `dataset.json` defines the dataset metadata:
```json
{
  "channel_names": { "0": "CT" },
  "labels": {
    "background": 0,
    "Bowel Wall Thickening": 1
  },
  "numTraining": 19,
  "file_ending": ".nii.gz"
}
```

- `labels.csv` Contains ground-truth classification labels for bowel wall thickening: 0 = diverticulitis, 1 = cancer:
```csv
UID,img_path,target
412,/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/imagesTr/412_0000.nii.gz,0
230,/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/imagesTr/230_0000.nii.gz,1
```

### Training Pipeline

#### 1) Generate colon masks with TotalSegmentator
These masks are used to crop CT scans during nnU-Net preprocessing.

- Install TotalSegmentator in a separate environment, then run:
```bash
python run_totalseg.py \
  --input /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/imagesTr \
  --output /data/colon_cancer/totalseg
```
- You may customize the `--input` and `--output` paths.

- Set the following environment variables:
```bash
export auto_seg_path="/data/colon_cancer/totalseg/total"            # colon annotations from TotalSegmentator
export nnUNet_raw="/data/colon_cancer/CC_Detection/raw_data"         # raw data
export nnUNet_preprocessed="/data/colon_cancer/CC_Detection/pp_data" # preprocessed data
export nnUNet_results="/data/colon_cancer/CC_Detection/nnUNet_results" # training logs/results
export root_path = "/data/colon_cancer/CC_Detection"
```

#### 2) Train segmentation and classification models

- nnU-Net preprocessing: nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```bash
nnUNetv2_plan_and_preprocess -d 100 -c 3d_fullres -crop_mode colon \
  --verify_dataset_integrity \
  -overwrite_target_spacing 1.0 1.0 1.0 \
  -overwrite_plans_name crop2colon_1spacing \
  -pl nnUNetPlannerResEncL
```
Notes:
- If `-c` is omitted, nnU-Net generates preprocessing for `2D`, `3d_fullres`, and `3d_lowres`.
- `-crop_mode`:
  - `colon`: crop around colon region using TotalSegmentator annotations
  - `label`: crop around ground truth labels
  - `none`: keep original spatial dimensions
- `-overwrite_target_spacing`: manually set resampling spacing
- `-overwrite_plans_name`: custom plan name when changing defaults
- `-pl`: planner type  
Preprocessed data for each configuration is saved under `nnUNet_preprocessed`.

- nnU-Net training: nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD :
```bash
nnUNetv2_train 100 3d_fullres 0 -p nnUNetResEncUNetLPlans -overwrite_plans_name crop2colon_1spacing --npz
```
- Use `-p` to match the planner used in preprocessing (if not default).
- Add `--npz` to store validation-set probability maps (needed for ensembling/best-config search).
- If `--npz` wasn’t used during training, you can later run:
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz
```

- ResNet classifier training:
```bash
train_resnet --config_file /data/benchaaben/ColonCancerDetection/classifier/run_config.json
```
Configurable parameters: patch size, batch size, patches per epoch, total epochs, optimizer, LR scheduler, model architecture, dataset paths.

- `pp_nnunet_data`: set to the plan name directory produced by nnU-Net (to consume its outputs). If `null`, the classifier will run its own preprocessing on raw data.
- `path_root_output`: directory for training logs and results.
- `path_root`: dataset root containing `imagesTr` and `labelsTr`.

Outputs:
- Checkpoints: `path_root_output/logs`
- MLflow logs: `path_root_output/mlruns`

### Inference

#### 3) Segmentation inference (select and run best configuration)
- Identify the best single configuration or ensemble based on cross-validation:
```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS
```
What it does:
- Evaluates the provided configurations (`-c`) and, by default, explores all 2-way ensembles.
- Requires `.npz` probability files from training with `--npz`.
- Disable ensembling with `--disable_ensembling` if desired.
- Automatically determines postprocessing (largest-component filtering for foreground/background and each label).

Outputs:
- Prints exact prediction commands to run.
- Creates in `nnUNet_results/DATASET_NAME`:
  - `inference_instructions.txt`: copy-pasteable prediction commands
  - `inference_information.json`: performance of configs/ensembles, postprocessing effect, and debug info

- Manual prediction example: nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```bash
nnUNetv2_predict \
  -d 100 \
  -i /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/imagesTs \
  -o /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer \
  -c 3d_fullres \
  -p nnUNetResEncUNetLPlans \
  -npp 1 -nps 1
```
- `-i`: input folder with raw test volumes
- `-o`: output folder for predicted segmentations (used by the classifier later)
- `-f`: folds to use
- `-c`: configuration
- `-p`: planner/plans name to match training

#### Classifier inference
Once segmentation predictions for the test set are available, run the classifier to categorize bowel wall thickening into cancer vs diverticulitis:
```bash
predict_resnet --config_file /data/benchaaben/ColonCancerDetection/classifier/run_config.json
```
- Adjust inference parameters in the config_file (e.g., patch size, overlap).
- Set:
  - `path_root_output`: directory to save results
  - `path_root`: dataset root where `imagesTs` and `labelsTs` reside
- Outputs:
  - Classification report: `path_root_output/checkpoint_ref/classification_report.txt`
  - Detailed results: `path_root_output/checkpoint_ref/results.csv`





