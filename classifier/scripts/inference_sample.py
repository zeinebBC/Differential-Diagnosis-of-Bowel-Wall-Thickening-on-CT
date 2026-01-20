from pathlib import Path
import torch
import torchio as tio
from collections import Counter
from classifier.models import ResNet
from classifier.data.utils.functions_utils import pad_batch_with_channel
import json
from classifier.data.utils.cropping import crop_to_label_region
from classifier.data.utils.resampling import resample_image
from classifier.data.utils.normalizing import window_and_normalize
import SimpleITK as sitk
import argparse


def get_model(config):
    if config["model"]["type"] == "ResNet":
        return ResNet(
            in_ch=config["model"]["in_ch"],
            out_ch=config["model"]["out_ch"],
            model=config["model"].get("resnet_model", 18),
            kwargs_resnet=config["model"].get("kwargs_resnet", {}),
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['type']}")


def run_pred(model, img, use_softmax=True):
    # Optionally check for supported models
    if not isinstance(model, ResNet):
        raise ValueError(f"Unknown model type: {type(model)}")

    # Forward pass
    pred = model(img)
    num_classes = pred.shape[1] if pred.ndim > 1 else 1

    if use_softmax:
        if num_classes == 1:
            pred = torch.sigmoid(pred)
        else:
            pred = torch.softmax(pred, dim=1)

    return pred


def load_sample(img_path, seg_path):
    # Read original images with SimpleITK to get metadata
    img_sitk_orig = sitk.ReadImage(str(img_path))
    lbl_sitk_orig = sitk.ReadImage(str(seg_path))

    # Convert to NumPy to crop
    img_np = sitk.GetArrayFromImage(img_sitk_orig)  # shape: [D,H,W]
    lbl_np = sitk.GetArrayFromImage(lbl_sitk_orig)

    # Crop to label region
    spacing = img_sitk_orig.GetSpacing()
    img_np, lbl_np, _ = crop_to_label_region(img_np, lbl_np, lbl_np, spacing)

    # Convert cropped NumPy back to SimpleITK using original metadata
    def np_to_sitk(cropped_np, reference_sitk):
        sitk_img = sitk.GetImageFromArray(cropped_np)
        sitk_img.SetOrigin(reference_sitk.GetOrigin())
        sitk_img.SetSpacing(reference_sitk.GetSpacing())
        sitk_img.SetDirection(reference_sitk.GetDirection())
        return sitk_img

    img_sitk_cropped = np_to_sitk(img_np, img_sitk_orig)
    lbl_sitk_cropped = np_to_sitk(lbl_np, lbl_sitk_orig)

    # Resample
    img_resampled = resample_image(img_sitk_cropped, (1, 1, 1), is_label=False)
    lbl_resampled = resample_image(lbl_sitk_cropped, (1, 1, 1), is_label=True)

    # Convert back to NumPy
    img_np = sitk.GetArrayFromImage(img_resampled)
    lbl_np = sitk.GetArrayFromImage(lbl_resampled)

    # Normalize
    img_np = window_and_normalize(img_np)

    # Reorder axes for PyTorch
    # img_np = np.transpose(img_np, (2, 1, 0))
    # lbl_np = np.transpose(lbl_np, (2, 1, 0))
    """
    debug_image_dir = Path("image")
    debug_label_dir = Path("label")
    debug_image_dir.mkdir(exist_ok=True)
    debug_label_dir.mkdir(exist_ok=True)
    
    # Convert tensors back to sitk for saving
    img_sitk_save = sitk.GetImageFromArray(img_np.transpose(2,1,0))  
    img_sitk_save.SetSpacing(img_resampled.GetSpacing())
    img_sitk_save.SetOrigin(img_resampled.GetOrigin())
    img_sitk_save.SetDirection(img_resampled.GetDirection())
    sitk.WriteImage(img_sitk_save, str(debug_image_dir / img_path.name))
    
    lbl_sitk_save = sitk.GetImageFromArray(lbl_np.transpose(2,1,0))
    lbl_sitk_save.SetSpacing(lbl_resampled.GetSpacing())
    lbl_sitk_save.SetOrigin(lbl_resampled.GetOrigin())
    lbl_sitk_save.SetDirection(lbl_resampled.GetDirection())
    sitk.WriteImage(lbl_sitk_save, str(debug_label_dir / seg_path.name))
    """
    # Convert to tensor
    img_t = torch.from_numpy(img_np).unsqueeze(0).float()
    lbl_t = torch.from_numpy(lbl_np).unsqueeze(0).float()
    # print(img_t.shape, lbl_t.shape)
    return torch.cat([img_t, lbl_t], dim=0).unsqueeze(0)


def run_inference_on_folder(input_folder, segmentation_folder):
    config_file = Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    chkpt_folder = config["testing"].get("chkpt_folder", "")
    patch_size = config["testing"].get("patch_size", [64, 256, 256])
    patch_overlap = config["testing"].get("patch_overlap", [32, 128, 128])
    aggregation_mode = config["testing"].get("aggregation_mode", "average")
    use_last = config["testing"].get("use_last", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # --- Initialize model ---
    model = get_model(config)
    if use_last:
        model = model.load_last_checkpoint(chkpt_folder)
    else:
        model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    # --- Inference ---
    input_folder = Path(input_folder)
    segmentation_folder = Path(segmentation_folder)

    predictions = {}
    for img_file in sorted(input_folder.glob("*.nii*")):
        uid = img_file.name.replace("_0000.nii.gz", "")
        seg_file = segmentation_folder / f"{uid}.nii.gz"

        if not seg_file.exists():
            print(f"[WARNING] No segmentation for {uid}")
            continue

        image = load_sample(img_file, seg_file)

        image = image.to(device)
        image = pad_batch_with_channel(image, patch_size, constant_values=0)[0]

        subject = tio.Subject(image=tio.ScalarImage(tensor=image))
        sampler = tio.inference.GridSampler(
            subject, patch_size=patch_size, patch_overlap=patch_overlap
        )

        patch_preds_sum = None
        patch_class_votes = []
        n_patches = 0

        for patches_batch in sampler:
            patches = patches_batch["image"][tio.DATA].to(device)
            patches = patches.unsqueeze(0)

            preds = run_pred(model, patches, use_softmax=True)
            preds_cpu = preds.detach().cpu()
            num_classes = preds_cpu.shape[1] if preds_cpu.ndim > 1 else 1

            if aggregation_mode == "average":
                patch_preds_sum = (
                    preds_cpu
                    if patch_preds_sum is None
                    else patch_preds_sum + preds_cpu
                )
            elif aggregation_mode == "majority":
                if num_classes == 1:
                    patch_classes = (preds_cpu > 0.5).int().flatten().tolist()
                else:
                    patch_classes = torch.argmax(preds_cpu, dim=1).flatten().tolist()
                patch_class_votes.extend(patch_classes)
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
            n_patches += 1

        # --- Final aggregation ---
        if aggregation_mode == "average":
            mean_prob = patch_preds_sum / n_patches
            if num_classes == 1:
                final_label = int((mean_prob > 0.5).item())
                mean_prob = mean_prob.flatten()
            else:
                final_label = torch.argmax(mean_prob).item()
        elif aggregation_mode == "majority":
            class_counts = Counter(patch_class_votes)
            final_label = max(class_counts, key=class_counts.get)
            mean_prob = torch.tensor(
                [class_counts[c] / n_patches for c in sorted(class_counts.keys())]
            )
        if num_classes == 1:
            predicted_prob = float(mean_prob[0].item())
        else:
            predicted_prob = float(mean_prob[0][final_label].item())

        predictions[uid] = {"disease": final_label, "probability": predicted_prob}

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run inference on a folder of images")
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Path to input images folder"
    )
    parser.add_argument(
        "--segmentation_folder",
        type=str,
        required=True,
        help="Path to segmentation folder",
    )
    args = parser.parse_args()

    pred = run_inference_on_folder(args.input_folder, args.segmentation_folder)
    print(pred)


if __name__ == "__main__":
    main()
