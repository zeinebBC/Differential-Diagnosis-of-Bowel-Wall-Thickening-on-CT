# An Automated Two-Stage Deep Learning Pipeline for Segmentation and Differential Diagnosis of Bowel Wall Thickening on CT Scans

This repository contains the official implementation of a fully automated diagnostic framework designed to localize and differentiate Bowel Wall Thickening (BWT) on 3D abdominal CT scans. The pipeline addresses a critical clinical challenge: distinguishing between **Colon Cancer** and **Diverticulitis**, two conditions with overlapping imaging features that require fundamentally different treatments.

---

## 📌 Project Overview

The system operates as a two-stage pipeline to eliminate the need for manual region-of-interest (ROI) definition:

* **Segmentation Stage:** Utilizes **nnU-Net v2** [[1]](#ref-nnunet) to automatically segment regions of pathological bowel wall thickening from 3D CT volumes.
* **Classification Stage:** Employs a dual-channel 3D ResNet-18 that processes the original CT volume concatenated with the predicted segmentation mask.


### Key Performance Metrics

* **Dice Score** 0.78 on internal test set (n=77)
* **Accuracy:** 91% on internal test set (n=77) (improved from 88% using CT-only input).
* **Sensitivity:** 98% for malignant pathology.
* **Specificity:** 81% for benign pathology.
* **Generalizability:** Achieved 98% accuracy on external validation set (Medical Decathlon).

---
## ⚖️ Pretrained Models 

Pretrained weights for both the segmentation and classification models will be publicly released upon publication of the associated manuscript.

## 📂 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Dataset Structure](#-dataset-structure)
4. [Training Pipeline](#-training-pipeline)
5. [Inference](#-inference)
6. [References](#-references)

---

## 🛠 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (NVIDIA H100L-94C used for training) 
- Conda or pip

### Step 1: Create Environment

```bash
conda create -n bwt_env python=3.10
conda activate bwt_env
```

### Step 2: Install PyTorch

Install PyTorch with CUDA support (adjust CUDA version based on your system):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu13
```

### Step 3: Install Package

Clone the repository and install the project:

```bash
git clone <repository-url>
cd ColonCancerDetection
pip install -e .
```
## 🚀 Quick Start

### 1. Set Environment Variables

```bash
export root="/path/to/your/data"
export nnUNet_raw="/path/to/raw/data"
export nnUNet_preprocessed="/path/to/preprocessed/data"
export nnUNet_results="/path/to/output/folder"
export auto_seg="/path/to/Totalsegmentator/annotations" 
```

### 2. Prepare Dataset

Organize your dataset following the [Dataset Structure](#dataset-structure) section.

### 3. Run Pipeline

Proceed to the Training Pipeline (#Training Pipeline) and Inference(#Inference) for step-by-step model execution.

## 📊 Dataset Structure

### Standard Structure (nnU-Net Style)

The pipeline supports datasets following nnU-Net conventions:

```
root_path/
├── pp_data/                    # Preprocessed data (auto-generated)
│   └── Dataset100_CC/
├── raw_data/                   # Raw dataset
│   └── Dataset100_CC/
│       ├── dataset.json        # Dataset metadata
│       ├── imagesTr/           # Training images
│       │   ├── 1_0000.nii.gz
│       │   ├── 2_0000.nii.gz
│       │   └── ...
│       ├── imagesTs/           # Test images
│       │   ├── 4_0000.nii.gz
│       │   └── ...
│       ├── labelsTr/           # Training labels (segmentation masks)
│       │   ├── 1.nii.gz
│       │   ├── 2.nii.gz
│       │   └── ...
│       ├── labelsTs/           # Test labels (optional, for evaluation)
│       │   └── ...
│       ├── labels.csv          # Classification labels (required)
│       └── splits.csv          # Train/val/test splits (auto-generated)
├── nnUNet_results/             # nnU-Net training outputs
│   └── Dataset100_CC/
└── Resnet_results/             # ResNet classifier training outputs

```

### Required Files

#### `dataset.json`

Dataset metadata for nnU-Net:

```json
{
  "channel_names": { "0": "CT" },
  "labels": {
    "background": 0,
    "Bowel Wall Thickening": 1
  },
  "numTraining": 735,
  "file_ending": ".nii.gz"
}
```

#### `labels.csv`

Classification labels for each case. Required columns: `UID`, `target`.

```csv
UID,target
412,0
230,1
555,0
```

- `UID`: Unique identifier matching the image filename (without `_0000.nii.gz`)
- `target`: Classification label
  - `0` = diverticulitis
  - `1` = cancer

#### `splits.csv`

Train/validation/test splits. **Auto-generated** during preprocessing, or provide your own:

```csv
UID,target,Split
412,0,train
230,1,val
555,0,test
```


## 🧠 Training Pipeline

### Step 1:  Anatomical Cropping (TotalSegmentator)

Use pretrained **TotalSegmentator** model [[2]](#ref-totalseg to generate colon masks for cropping CT scans during nnU-Net preprocessing. This reduces computational load and background dominance. 

When to run: Before both training and inference pipelines.

```bash
python Differential-Diagnosis-of-BWT/run_totalseg.py \
  --input /path/to/images \
  --output /path/to/Totalsegmentator/annotations \
  --task total \
  --roi_subset colon
```

### Step 2: Train Segmentation Model (nnU-Net)

#### 2.1 Preprocessing

generate the model's configuration and training plan, preprocess the dataset:

```bash
nnUNetv2_plan_and_preprocess \
  -d 100 \
  -c 3d_fullres \
  -crop_mode colon \
  --verify_dataset_integrity \
  -overwrite_target_spacing 1.0 1.0 1.0 \
  -pl nnUNetPlannerResEncL
```

**Arguments:**
- `-d`: Dataset ID or name (e.g., `100` or `Dataset100`)
- `-c`: Configuration (`2d`, `3d_fullres`, `3d_lowres`, or omit for all)
- `-crop_mode`: Cropping strategy
  - `colon`: Crop around colon region (uses TotalSegmentator masks)
  - `label`: Crop around ground truth labels
  - `none`: No cropping
- `--verify_dataset_integrity`: Run sanity checks
- `-overwrite_target_spacing`: Set voxel spacing (e.g., `1.0 1.0 1.0`)
- `-pl`: Planner type

**Output:** Preprocessed data saved to `nnUNet_preprocessed/{Dataset_name}/`

#### 2.2 Training

Train the segmentation model:

```bash
nnUNetv2_train \
  100 \
  3d_fullres \
  0 \
  -p nnUNetResEncUNetLPlans \
  --npz
```

**Arguments:**
- `100`: Dataset ID or name
- `3d_fullres`: nnUNet Configuration
- `0`: Fold number (0-4 for 5-fold cross-validation)
- `-p`: Planner name (must match preprocessing)
- `--npz`: Save probability maps (required for model ensembling)



**Generate probability maps later if needed:**
```bash
nnUNetv2_train 100 3d_fullres 0 \
  -p nnUNetResEncUNetLPlans \
  --val --npz
```

### Step 3: Train Classification Model (ResNet)

#### 3.1 Configure Training

Edit `classifier/run_config.json`:

```json
{
  "model": {
    "type": "ResNet",
    "in_ch": 2,
    "out_ch": 2,
    "resnet_model": 18,
    "pretrained": false
  },
  "training": {
    "dataset": "Dataset100_CC",
    "patch_size": [32, 156, 156],
    "batch_size": 8,
    "num_epochs": 100,
    "num_patches_per_epoch_train": 2500,
    "num_patches_per_epoch_val": 250,
    "resample_spacing": [1.0, 1.0, 1.0],
    "output_dir": "Resnet_results",
    "dual_input": true,
    "use_gt": false,
  }
}
```

**Key parameters:**
- `in_ch`: number of input channels
- `out_ch`: number of output channels (=classes)
- `dataset`: Dataset name
- `patch_size`: Patch dimensions `[D, H, W]`
- `batch_size`: Batch size
- `num_epochs`: Training epochs
- `output_dir`: Output directory (relative to `root_path`)
- `dual_input`: whether to add the segmentation masks as a second input channel
- `use_gt`: whether to use the ground truth segmentation masks or the nnU-Net predictions.

You can also configure additional parameters such as the optimizer and learning rate scheduler in the same configuration file.
#### 3.2 Run Training

```bash
train_resnet
```

**Outputs:**
- Checkpoints: `{root_path}/Resnet_results/logs/{model}_{dataset}_{timestamp}/`
- MLflow logs: `{root_path}/Resnet_results/mlruns/`

## Inference

### Step 1: Segmentation Inference

#### 1.1 Find Best Configuration

Evaluate configurations to find the best single model or ensemble (required for combining predictions from different trained configurations/folds)

```bash
nnUNetv2_find_best_configuration 100 
```

**Outputs:**
- `nnUNet_results/Dataset100/inference_instructions.txt`: copy-pasteable prediction commands
- `nnUNet_results/Dataset100/inference_information.json`: performance of configs/ensembles, postprocessing effect, and debug info

#### 1.2 Run Prediction

Use the commands from `inference_instructions.txt`, or run manually:

```bash
nnUNetv2_predict \
  -d 100 \
  -i /path/to/raw_data/Dataset100/imagesTs \
  -o /path/to/raw_data/Dataset100/predictionsTs \
  -f 0 1 2 3 4 \
  -tr nnUNetTrainer \
  -c 3d_fullres \
  -p nnUNetResEncUNetLPlans \
  -npp 1 \
  -nps 1 \
  --save_probabilities
```

**Arguments:**
- `-d`: Dataset ID or name
- `-i`: Input folder (test images)
- `-o`: Output folder (predictions)
- `-f`: Folds to use (e.g., `0 1 2 3 4`)
- `-tr`: nnUNet Trainer name
- `-c`: nnUNet Configuration
- `-p`: nnUNet Planner name
- `-npp`: Number of preprocessing processes
- `-nps`: Number of segmentation export processes
- `--save_probabilities`: Save probability maps (for debugging or post-processing)

### Step 2: Classification Inference
After segmentation predictions are available:
#### 2.1 Configure Inference

Edit `classifier/run_config.json` (testing section):

```json
{
  "testing": {
    "dataset": "Dataset100",
    "chkpt_folder":"ResNet_Dataset100_CC_2026_02_11_183514",
    "output_dir": "Resnet_results/inf_outputs",
    "patch_size": [32, 156, 156],
    "patch_overlap": [16, 64, 64],
    "aggregation_mode": "average",
    "use_gt": false,
    "dual_input": true,
  }
}
```

**Key parameters:**
- `chkpt_folder`: name of the trained model checkpoints.
- `patch_size`: Spatial dimensions of the input patches used during inference.
- `patch_overlap`: Amount of overlap between neighboring patches in sliding-window inference.
- `aggregation_mode`: Strategy used to combine patch-level predictions into a single prediction per sample:
  - `average`: Averages the prediction scores (e.g., probabilities) across all patches.
  - `majority`: Assigns the final label based on a majority vote of patch-level predictions.

#### 2.2 Run Inference

```bash
predict_resnet
```

**Outputs:**
- Classification report: `{root_path}/Resnet_results/inf_outputs/{checkpoint_id}/{dataset}/classification_report.txt`
- Detailed results per case: `{root_path}/Resnet_results/inf_outputs/{checkpoint_id}/{dataset}/results.csv`


## 📚 References

<a id="ref-nnunet"></a>
**[1] Isensee, F., et al.**  
*nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.*  
**Nature Methods**, 18, 203–211 (2021).  
https://doi.org/10.1038/s41592-020-01008-z 
Code: https://github.com/MIC-DKFZ/nnUNet

<a id="ref-totalseg"></a>
**[2] Wasserthal, J., et al.**  
*TotalSegmentator: robust segmentation of 104 anatomical structures in CT images.*  
**Radiology: Artificial Intelligence**, 5(5), e230024 (2023).  
https://doi.org/10.1148/ryai.230024
Code: https://github.com/wasserth/TotalSegmentator




