from pathlib import Path
import argparse
import json
import logging
import torch
import torchio as tio
from tqdm import tqdm
from models import ResNet
from data import ColonCancer
from scripts.utils.gradcam_utils import *
from data.utils.functions_utils import pad_to_shape


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help="Path to JSON config used for training/testing/gradcam")
    
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    gradcam_cfg = config.get("gradcam", {})

    chkpt_folder = gradcam_cfg.get("chkpt_folder", "")
    output_dir = gradcam_cfg.get("output_dir", "")
    use_labels = gradcam_cfg.get("use_labels", True)
    patch_size = gradcam_cfg.get("patch_size", [64, 256, 256])
    patch_overlap = gradcam_cfg.get("patch_overlap", [32, 128, 128])
    list_cases = gradcam_cfg.get("list_cases", None)
    target_class = gradcam_cfg.get("target_class", None)
    alpha = gradcam_cfg.get("alpha", 0.35)
    num_workers = gradcam_cfg.get("num_workers", 8)
    use_last = gradcam_cfg.get("use_last", False)
    path_root = gradcam_cfg.get("path_root",f"/data/coloncancer/Classifier")
    dataset_name = gradcam_cfg.get("dataset", None)

    # --- Setup output folder ---
    path_out = Path(output_dir) / Path(chkpt_folder).name
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    # Logging
    logger = logging.getLogger("gradcam")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # --- Load dataset ---
    ds_test = ColonCancer(
        dataset_name=dataset_name,
        patch_size=patch_size,
        split='test',
        return_full_image=True,
        path_root=path_root,
        use_labels=use_labels
    )

   
    # --- Load model ---
    model = get_model(config)
    if use_last:
        model = model.load_last_checkpoint(chkpt_folder)
    else:
        model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    # --- Grad-CAM setup ---
    target_module = find_target_module_for_resnet3d(model)
    cam_extractor = GradCAM3D(model, target_module, device=device)

    if not list_cases:
        raise ValueError("Please provide --list_cases in JSON config or CLI")

    # --- Iterate over cases ---
    for i, uid in enumerate(tqdm(list_cases, total=len(list_cases), desc="Cases")):
        logger.addHandler(logging.FileHandler(path_out / f'gradcam_log_{uid}.txt', mode='w'))

        image, label, target = load_case_from_dataset(ds_test, uid, in_ch=model.in_ch)
        image, label = image.to(device), label.to(device)
        image = pad_to_shape(image, tuple(patch_size))
        label = pad_to_shape(label, tuple(patch_size))

        torch.cuda.synchronize() if device.type == 'cuda' else None

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image[0].detach().cpu()),
            label=tio.LabelMap(tensor=label[0].detach().cpu())
        )

        sampler = tio.inference.GridSampler(subject, patch_size=patch_size, patch_overlap=patch_overlap)
        out_case_dir = path_out / f"{uid}"
        out_case_dir.mkdir(parents=True, exist_ok=True)

        for patch_idx, patch in enumerate(sampler):
            patch_img = patch['image'][tio.DATA].unsqueeze(0).to(device)
            patch_lbl = patch['label'][tio.DATA].unsqueeze(0).to(device)
            location = patch['location']

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

            torch.set_grad_enabled(True)
            cam = cam_extractor(patch_img, target_class=target_class if target_class is not None else pred)
            torch.set_grad_enabled(False)

            cam_cpu = cam.detach().cpu()
            vol_cpu = patch_img.detach().cpu()
            lbl_cpu = patch_lbl.detach().cpu()

            cam_nii = out_case_dir / f"patch{patch_idx:04d}_cam_pred{pred}_p{prob:.3f}.nii.gz"
            save_cam_nifti(cam_cpu, cam_nii)

            out_png = out_case_dir / f"patch{patch_idx:04d}_overlay_pred{pred}_p{prob:.3f}.png"
            overlay_and_save_slices(vol_cpu, cam_cpu, lbl_cpu, out_png, alpha=alpha)

            logger.info(f"uid={uid} patch={patch_idx} loc={location} target={int(target.item())} pred={pred} prob={prob:.3f} saved={cam_nii.name},{out_png.name}")

    cam_extractor.remove_hooks()
    print(f"Saved Grad-CAM outputs under: {path_out}")


if __name__ == "__main__":
    main()
