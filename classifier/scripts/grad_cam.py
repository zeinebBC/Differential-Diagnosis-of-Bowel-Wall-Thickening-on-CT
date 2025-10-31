# gradcam_pipeline.py
import torch
from pathlib import Path

import json
import logging
from tqdm import tqdm
import torchio as tio

import os
from classifier.models import ResNet
from classifier.data import ColonCancer
from classifier.scripts.utils.gradcam_utils import *
from classifier.data.utils.functions_utils import pad_to_shape


def get_model(config):
    """Initialize ResNet from config"""
    if config["model"]["type"] == 'ResNet':
        return ResNet(
            in_ch=config["model"]["in_ch"],
            out_ch=config["model"]["out_ch"],
            model=config["model"].get("resnet_model", 18),
            kwargs_resnet=config["model"].get("kwargs_resnet", {}),
        )
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")


def main():
    
    config_file =Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    path_root = Path(os.environ.get("root_path"))
    gradcam_cfg = config.get("gradcam", {})
    chkpt_folder = gradcam_cfg.get("chkpt_folder", "")
    output_dir = path_root / Path(gradcam_cfg.get("output_dir", "")) / Path(chkpt_folder).name
    output_dir.mkdir(parents=True, exist_ok=True)
    list_cases = gradcam_cfg.get("list_cases", None)
    target_class = gradcam_cfg.get("target_class", None)
    alpha = gradcam_cfg.get("alpha", 0.35)
    patch_size = gradcam_cfg.get("patch_size", [64, 256, 256])
    patch_overlap = gradcam_cfg.get("patch_overlap", [32, 128, 128])
    
    dataset_name = gradcam_cfg.get("dataset", None)
    use_labels = gradcam_cfg.get("use_labels", True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    # Logging
    logger = logging.getLogger("gradcam")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    log_file = output_dir / 'gradcam_log.txt'
    logger.addHandler(logging.FileHandler(log_file, mode='a'))

    # --- Load dataset ---
    ds_test = ColonCancer(
        dataset_name=dataset_name,
        patch_size=patch_size,
        split='test',
        return_full_image=True,
        use_labels=use_labels
    )

    # --- Load model ---
    model = get_model(config)
    model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    # --- Grad-CAM setup ---
    target_module = find_target_module_for_resnet3d(model)
    cam_extractor = GradCAM3D(model, target_module, device=device)

    if not list_cases:
        raise ValueError("Please provide --list_cases in JSON config")

    # --- Storage for unnormalized CAMs ---
    all_cams = {}  # {uid: [(cam_tensor, image_tensor, label_tensor, patch_idx, pred, prob)]}

    # --- First pass: compute CAMs and update global min/max ---
    for uid in tqdm(list_cases, desc="Computing Grad-CAM"):
        sample = ds_test.get_item_by_uid(uid)
        image, label, target = sample["source"], sample["label"], sample["target"]
        image, label = image.to(device), label.to(device)
        image = pad_to_shape(image, tuple(patch_size))
        label = pad_to_shape(label, tuple(patch_size))

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image[0].detach().cpu()),
            label=tio.LabelMap(tensor=label[0].detach().cpu())
        )
        sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap)

        all_cams[uid] = []
        out_case_dir = output_dir / f"{uid}"
        out_case_dir.mkdir(parents=True, exist_ok=True)

        for patch_idx, patch in enumerate(sampler):
            patch_img = patch['image'][tio.DATA].unsqueeze(0).to(device)
            patch_lbl = patch['label'][tio.DATA].unsqueeze(0).to(device)

            # Forward to get prediction
            with torch.no_grad():
                logits = model(patch_img)
                if model.out_ch == 1:
                    probs = torch.sigmoid(logits)
                    pred = int((probs > 0.5).item())
                    prob = probs.item()
                else:
                    probs = torch.softmax(logits, dim=1)
                    pred = probs.argmax(dim=1).item()
                    prob = probs[0, pred].item()

            # Grad-CAM
            torch.set_grad_enabled(True)
            cam = cam_extractor(patch_img, target_class=target_class if target_class is not None else pred)
            torch.set_grad_enabled(False)

            # Store raw CAM for later normalization
            all_cams[uid].append((cam.detach().cpu(), patch_img.detach().cpu(), patch_lbl.detach().cpu(), patch_idx, pred, prob))

    # --- Second pass: visualize CAMs using final global min/max ---
    global_min = cam_extractor.global_min
    global_max = cam_extractor.global_max

    for uid, cam_list in tqdm(all_cams.items(), desc="Visualizing Grad-CAM"):
        out_case_dir = output_dir / f"{uid}"
        for cam, vol, lbl, patch_idx, pred, prob in cam_list:
            cam_nii = out_case_dir / f"patch{patch_idx:04d}_cam_pred{pred}_p{prob:.3f}.nii.gz"
            save_cam_nifti(cam, cam_nii)
        
            out_png = out_case_dir / f"patch{patch_idx:04d}_overlay_pred{pred}_p{prob:.3f}.png"
            overlay_and_save_slices(vol, cam, global_min, global_max, lbl, out_png, alpha=alpha)

            logger.info(f"uid={uid} patch={patch_idx} pred={pred} prob={prob:.3f} saved={cam_nii.name},{out_png.name}")

    cam_extractor.remove_hooks()
    print(f"Saved Grad-CAM outputs under: {output_dir}")


if __name__ == "__main__":
    main()
