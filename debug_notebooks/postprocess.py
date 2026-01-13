#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import nibabel as nib
import cc3d
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import seg_metrics.seg_metrics as sg
"""
# ============================================================
# Paths (defaults – can be overridden via CLI)
# ============================================================

DEFAULT_PRED_ROOT  = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp"
DEFAULT_GT_ROOT    = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs"
DEFAULT_COLON_ROOT = "/data/colon_cancer/totalseg/total"
DEFAULT_PROB_ROOT  = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp"
DEFAULT_OUT_DIR    = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_pp"
DEFAULT_JSON       = "metrics_results_postprocessing.json"

# ============================================================
# Verbose printing helper
# ============================================================

def vprint(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)

# ============================================================
# Loading / Saving
# ============================================================

def load_nifti(path):
    img = nib.load(str(path))
    return img.get_fdata(), img.affine, img.header

def load_label(path):
    return load_nifti(path)[0].astype(np.uint8)

def save_nifti(data, affine, header, path):
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
    vprint(verbose, f"    DEBUG: spacing (Z,Y,X) = {spacing}")
    return spacing

# ============================================================
# Post-processing
# ============================================================

def components_touching_colon(pred_mask, colon_mask, verbose=False):
    vprint(verbose, f"    DEBUG: components_touching_colon")
    labeled, num = cc3d.connected_components(pred_mask, return_N=True, connectivity=26)
    colon_mask_bool = colon_mask > 0
    keep_ids = [i for i in range(1, num + 1) if np.any((labeled == i) & colon_mask_bool)]
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

    z_ranges = [(np.where(c)[2].min(), np.where(c)[2].max()) for c in components if c.any()]

    for i, (c, (z0, z1)) in enumerate(zip(components, z_ranges)):
        if used[i]:
            continue
        m = c.copy()
        used[i] = True
        for j, (c2, (zz0, zz1)) in enumerate(zip(components, z_ranges)):
            if not used[j] and not (zz1 < z0 or zz0 > z1):
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

#METRICS = ["dice", "precision", "recall", "fpr", "fnr", "msd", "hd95"]
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

    pred_files = sorted(f for f in os.listdir(args.pred_root) if f.endswith((".nii", ".nii.gz")))

    if args.debug_case:
        pred_files = [f for f in pred_files if args.debug_case in f]

    for pred_file in pred_files:
        uid = pred_file.replace(".nii.gz", "").replace(".nii", "")
        print(f"\n▶ Processing {uid}")

        pred_path = Path(args.pred_root) / pred_file
        gt_path   = Path(args.gt_root) / pred_file

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

        results[uid] = {"before": metrics_before, "after": metrics_after}
        before_all.append(metrics_before)
        after_all.append(metrics_after)

        if args.debug_case:
            break

    summary = {
        "mean_before": {m: float(np.mean([x[m] for x in before_all])) for m in METRICS},
        "mean_after":  {m: float(np.mean([x[m] for x in after_all])) for m in METRICS},
    }

    with open(args.output_json, "w") as f:
        json.dump({"per_case": results, "summary": summary}, f, indent=4)

    print("\n✅ Done")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", default=DEFAULT_PRED_ROOT)
    parser.add_argument("--gt-root", default=DEFAULT_GT_ROOT)
    parser.add_argument("--colon-root", default=DEFAULT_COLON_ROOT)
    parser.add_argument("--prob-root", default=DEFAULT_PROB_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-json", default=DEFAULT_JSON)

    parser.add_argument("--use-colon-filter", action="store_true")
    parser.add_argument("--use-interpolation", action="store_true")
    parser.add_argument("--merge-by-z", action="store_true")
    parser.add_argument("--final-selection", choices=["largest", "highest_score"], default="highest_score")
    parser.add_argument("--save-masks", action="store_true")

    parser.add_argument("--debug-case", type=str)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    evaluate(args)



nohup python evaluate_postprocessing.py \
  --pred_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp \
  --gt_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs \
  --colon_root /data/colon_cancer/totalseg/total \
  --prob_root /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_wp \
  --out_mask_dir /data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_pp \
  --output_json metrics_results_postprocessing_testingcode.json \
  --use_colon_filter \
  --use_interpolation \
  --final_selection highest_score \
  --merge_by_z \
  > postprocessing.log 2>&1 &
"""

#!/usr/bin/env python3
import os
import json
import numpy as np
import nibabel as nib
import seg_metrics.seg_metrics as sg

# -----------------------------
# Configuration
# -----------------------------
pred_dir = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/predictionsTs_pp"
gt_dir   = "/data/colon_cancer/CC_Detection/raw_data/Dataset100_CC/labelsTs"

output_file = "metrics_results_Ts_pp.json"

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
    "msd",
    "hd95",
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
        f for f in os.listdir(pred_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    )

    for pf in pred_files:
        pred_path = os.path.join(pred_dir, pf)
        gt_path   = os.path.join(gt_dir, pf)

        if not os.path.exists(gt_path):
            print(f"⚠️ GT not found for {pf}, skipping")
            continue

        pred, _ = load_nifti(pred_path)
        gt, _   = load_nifti(gt_path)

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
        "mean":   {m: float(np.mean(all_metrics[m])) for m in METRICS},
        "median": {m: float(np.median(all_metrics[m])) for m in METRICS},
        "min":    {m: float(np.min(all_metrics[m])) for m in METRICS},
        "max":    {m: float(np.max(all_metrics[m])) for m in METRICS},
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
