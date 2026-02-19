#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
import seg_metrics.seg_metrics as sg
from scipy.ndimage import distance_transform_edt

# ============================================================
# Paths (defaults – can be overridden via CLI)
# ============================================================

DEFAULT_PRED_ROOT = (
    "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp"
)
DEFAULT_GT_ROOT = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs"
DEFAULT_COLON_ROOT = "/data/colon_cancer/totalseg/total"
DEFAULT_PROB_ROOT = (
    "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp"
)
DEFAULT_OUT_DIR = (
    "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_pp"
)
DEFAULT_JSON = "metrics_results_postprocessing.json"

# ============================================================
# Verbose printing helper
# ============================================================


def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs, flush=True)


# ============================================================
# Loading / Saving
# ============================================================


def load_nifti(path):
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header


def load_label(path):
    return load_nifti(path)[0].astype(np.uint8)


def save_nifti(data, affine, header, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), str(path))


def load_fg_probs(npz_path):
    return np.transpose(np.load(npz_path)["probabilities"][1], (2, 1, 0))


# ============================================================
# Spacing helper
# ============================================================


def get_spacing(gt_path, pred_path, verbose=False):
    if gt_path.exists():
        hdr = nib.load(str(gt_path)).header
    else:
        hdr = nib.load(str(pred_path)).header

    spacing = hdr.get_zooms()[:3]
    vprint(verbose, f"    DEBUG: spacing (X,Y,Z) = {spacing}")
    return spacing


# ============================================================
# Post-processing
# ============================================================


def components_touching_colon(pred_mask, colon_mask, verbose=False):
    vprint(verbose, f"    DEBUG: components_touching_colon")
    vprint(
        verbose,
        f"    DEBUG: pred_mask shape = {pred_mask.shape}, colon_mask shape = {colon_mask.shape}",
    )
    labeled, num = cc3d.connected_components(pred_mask, return_N=True, connectivity=26)
    colon_mask_bool = colon_mask > 0
    keep_ids = [
        i for i in range(1, num + 1) if np.any((labeled == i) & colon_mask_bool)
    ]
    vprint(verbose, f"    DEBUG: keeping {len(keep_ids)} / {num} components")
    return np.isin(labeled, keep_ids).astype(np.uint8)


def interpolate_missing_slices(mask, verbose=False):
    vprint(verbose, f"    DEBUG: interpolate_missing_slices, shape={mask.shape}")
    mask = mask.astype(bool)

    z_any = mask.any(axis=(0, 1))
    z_idx = np.where(z_any)[0]

    if len(z_idx) < 2:
        vprint(verbose, "    DEBUG: <2 slices → skip interpolation")
        return mask.astype(np.uint8)

    dt = np.zeros_like(mask, dtype=np.float32)
    for z in range(mask.shape[2]):
        fg = mask[:, :, z]
        dt[:, :, z] = distance_transform_edt(~fg) - distance_transform_edt(fg)

    repaired = mask.copy()
    for z in range(z_idx[0] + 1, z_idx[-1]):
        if not z_any[z]:
            zp = z_idx[z_idx < z][-1]
            zn = z_idx[z_idx > z][0]
            w = (z - zp) / float(zn - zp)
            repaired[:, :, z] = ((1 - w) * dt[:, :, zp] + w * dt[:, :, zn]) < 0

    return repaired.astype(np.uint8)


def connected_components(mask):
    labeled = cc3d.connected_components(mask)
    return [(labeled == cid) for cid in range(1, labeled.max() + 1)]


def merge_components_by_z_overlap(components, verbose=False):
    vprint(verbose, f"    DEBUG: merging {len(components)} components by z-overlap")
    merged, used = [], [False] * len(components)

    z_ranges = [
        (np.where(c)[2].min(), np.where(c)[2].max()) for c in components if c.any()
    ]

    for i, (c, (z0, z1)) in enumerate(zip(components, z_ranges)):
        if used[i]:
            continue
        m = c.copy()
        used[i] = True
        for j, (c2, (zz0, zz1)) in enumerate(zip(components, z_ranges)):
            if not used[j] and not (zz1 <= z0 or zz0 >= z1):
                m |= c2
                used[j] = True
        merged.append(m)

    return merged


def keep_largest_component(mask):
    labeled = cc3d.connected_components(mask)
    if labeled.max() == 0:
        return mask
    sizes = cc3d.statistics(labeled)["voxel_counts"]
    return (labeled == (np.argmax(sizes[1:]) + 1)).astype(np.uint8)


def keep_highest_score_component(mask, fg_probs, merge_by_z, verbose=False):
    if not mask.any():
        vprint(verbose, "    DEBUG: empty mask → skipping highest-score selection")
        return mask.astype(np.uint8)

    comps = connected_components(mask)
    if merge_by_z and len(comps) > 1:
        comps = merge_components_by_z_overlap(comps, verbose)
    scores = [fg_probs[c].mean() if c.any() else -np.inf for c in comps]
    best = np.argmax(scores)
    vprint(verbose, f"    DEBUG: component scores = {scores}")
    return comps[best].astype(np.uint8)


# ============================================================
# Metrics
# ============================================================

# METRICS = ["dice", "precision", "recall", "fpr", "fnr", "msd", "hd95"]
METRICS = ["dice", "precision", "recall"]


def compute_metrics(pred, gt, spacing, verbose=False):
    vprint(verbose, f"    DEBUG: computing metrics")
    metrics = sg.write_metrics(
        labels=[1],
        pred_img=pred.astype(np.uint8),
        gdth_img=gt.astype(np.uint8),
        metrics=METRICS,
        spacing=spacing,
    )
    return {m: float(metrics[0][m][0]) for m in METRICS}


# ============================================================
# Main evaluation
# ============================================================


def evaluate(args):
    results = {}
    before_all, after_all = [], []

    pred_files = sorted(
        f for f in os.listdir(args.pred_root) if f.endswith((".nii", ".nii.gz"))
    )

    if args.debug_case:
        pred_files = [f for f in pred_files if args.debug_case in f]

    for pred_file in pred_files:
        uid = pred_file.replace(".nii.gz", "").replace(".nii", "")
        print(f"\n▶ Processing {uid}")

        pred_path = Path(args.pred_root) / pred_file
        gt_path = Path(args.gt_root) / pred_file

        pred, aff, hdr = load_nifti(pred_path)
        gt = load_label(gt_path)
        spacing = get_spacing(gt_path, pred_path, args.verbose)

        metrics_before = compute_metrics(pred, gt, spacing, args.verbose)

        m = pred.copy()

        if args.use_colon_filter:
            colon = load_label(Path(args.colon_root) / f"{uid}.nii.gz")
            m = components_touching_colon(m, colon, args.verbose)

        if args.use_interpolation:
            m = interpolate_missing_slices(m, args.verbose)

        if args.final_selection == "largest":
            m = keep_largest_component(m)
        else:
            fg_probs = load_fg_probs(Path(args.prob_root) / f"{uid}.npz")
            m = keep_highest_score_component(m, fg_probs, args.merge_by_z, args.verbose)

        metrics_after = compute_metrics(m, gt, spacing, args.verbose)

        if args.save_masks:
            save_nifti(m, aff, hdr, Path(args.out_dir) / pred_file)
            print(f"    Saved post-processed mask to {Path(args.out_dir) / pred_file}")

        results[uid] = {"before": metrics_before, "after": metrics_after}
        before_all.append(metrics_before)
        after_all.append(metrics_after)

        if args.debug_case:
            break

    summary = {
        "mean_before": {m: float(np.mean([x[m] for x in before_all])) for m in METRICS},
        "mean_after": {m: float(np.mean([x[m] for x in after_all])) for m in METRICS},
    }

    with open(args.output_json, "w") as f:
        json.dump({"per_case": results, "summary": summary}, f, indent=4)

    print("\n✅ Done")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--gt_root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--colon_root", default=DEFAULT_COLON_ROOT)
    parser.add_argument("--prob_root", default=DEFAULT_PROB_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_json", default=DEFAULT_JSON)

    parser.add_argument("--use_colon_filter", action="store_true")
    parser.add_argument("--use_interpolation", action="store_true")
    parser.add_argument("--merge_by_z", action="store_true")
    parser.add_argument(
        "--final_selection",
        choices=["largest", "highest_score"],
        default="highest_score",
    )
    parser.add_argument("--save_masks", action="store_true")

    parser.add_argument("--debug_case", type=str)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    evaluate(args)


"""
nohup python run_postprocessing.py \
  --pred_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp \
  --gt_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs \
  --colon_root /data/colon_cancer/totalseg/total \
  --prob_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp \
  --out_dir /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_pp \
  --output_json metrics_results_postprocessing_testingcode.json \
  --use_colon_filter \
  --use_interpolation \
  --final_selection highest_score \
  --merge_by_z \
  > postprocessing.log 2>&1 &


  nohup python run_postprocessing.py   --pred_root /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsDecath   --gt_root /data/colon_cancer/CC_Detection/raw_data/Decathlon/labelsTs   --colon_root /data/colon_cancer/totalseg/outputs_decathlon/total   --prob_root /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsDecath   --out_dir /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsDecathlon_pp   --output_json /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/results_decathlon_pp.json   --use_colon_filter   --final_selection largest   --verbose   > postprocessing_Decathlon.log 2>&1 &
  nohup python run_postprocessing.py   --pred_root /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsStage2   --gt_root /data/colon_cancer/CC_Detection/raw_data/stage_2_cc/labelsTs   --colon_root /data/colon_cancer/totalseg/outputs_cc2stage/total   --prob_root /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsStage2   --out_dir /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/predictionsStage2_pp   --output_json /data/colon_cancer/CC_Detection/raw_data/Dataset109_CC/final/results_stage2_pp.json   --use_colon_filter   --final_selection largest   --verbose   > postprocessing_Stage2.log 2>&1 &
"""
