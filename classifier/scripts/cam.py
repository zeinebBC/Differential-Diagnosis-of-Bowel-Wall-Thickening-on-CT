# cam_pipeline.py

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt

from classifier.models import ResNet
from classifier.data import ColonCancer
from classifier.data.utils.functions_utils import pad_to_shape


# ------------------------- Utilities -------------------------

def get_model(config: Dict) -> ResNet:
    if config["model"]["type"] == 'ResNet':
        return ResNet(
            in_ch=config["model"]["in_ch"],
            out_ch=config["model"]["out_ch"],
            model=config["model"].get("resnet_model", 18),
            kwargs_resnet=config["model"].get("kwargs_resnet", {})
        )
    raise ValueError(f"Unknown model type: {config['model']['type']}")


def find_last_conv_layer4(model: ResNet) -> torch.nn.Module:
    # Hook last conv in layer4 (ResNet3D MONAI)
    return model.model.layer4[-1].conv2


def save_cam_nifti(cam: torch.Tensor, out_path: Path) -> None:
    # cam expected [1, 1, D, H, W] on CPU
    cam_data = cam.squeeze(0) if cam.ndim == 5 else cam
    tio.ScalarImage(tensor=cam_data).save(str(out_path))


def overlay_and_save_slices(
    volume: torch.Tensor,
    cam: torch.Tensor,
    min_v: float,
    max_v: float,
    label: Optional[torch.Tensor],
    out_png: Path,
    alpha: float = 0.20,
    num_slices: int = 6,
) -> None:
    vol = volume.squeeze().numpy()
    heat = cam.squeeze().numpy()
    heat = (heat - min_v) / (max_v - min_v + 1e-6)
    lbl = label.squeeze().numpy() if label is not None else None

    if vol.ndim == 4:
        vol = vol[0, ...]
    D, W, H = vol.shape

    z_slices = np.linspace(0, D - 1, num_slices, dtype=int)
    rows = 3 if label is not None else 2
    fig, axes = plt.subplots(rows, num_slices, figsize=(2.5 * num_slices, 2.5 * rows))
    if num_slices == 1:
        axes = np.array([[axes]]) if rows == 1 else np.array([axes])

    for i, z in enumerate(z_slices):
        ax = axes[0, i] if rows > 1 else axes[i]
        ax.imshow(vol[z, :, :], cmap="gray")
        ax.set_title(f"z={z}")
        ax.axis("off")

    for i, z in enumerate(z_slices):
        ax = axes[1, i] if rows > 1 else axes[i]
        ax.imshow(vol[z, :, :], cmap="gray")
        ax.imshow(heat[z, :, :], cmap="jet", alpha=alpha, vmin=0, vmax=1)
        ax.axis("off")

    if label is not None:
        for i, z in enumerate(z_slices):
            ax = axes[2, i]
            ax.imshow(vol[z, :, :], cmap="gray")
            ax.imshow(np.ma.masked_where(lbl[z, :, :] == 0, lbl[z, :, :]), cmap="autumn", alpha=0.35)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


# ------------------------- Base and Implementations -------------------------

class BaseCAM3D:
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module, device: torch.device):
        self.model = model
        self.target_module = target_module
        self.device = device



    @torch.no_grad()
    def _resize_like_input(self, cam: torch.Tensor, input_3d: torch.Tensor) -> torch.Tensor:
        return F.interpolate(cam, size=input_3d.shape[-3:], mode="trilinear", align_corners=False)


    def compute(self, x: torch.Tensor, class_index: int) -> torch.Tensor:
        raise NotImplementedError

    def remove_hooks(self) -> None:
        pass


class GradCAM3D(BaseCAM3D):
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module, device: torch.device):
        super().__init__(model, target_module, device)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fwd_handle = None
        self.bwd_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.fwd_handle = self.target_module.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_module.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()

    def compute(self, x: torch.Tensor, class_index: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        logits = self.model(x)  # [B, C]
        is_binary = logits.shape[1] == 1

        if is_binary:
            score = logits[:, 0].sum()
        else:
            score = logits[:, class_index].sum()

        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_module selection.")

        weights = self.gradients.mean(dim=(-3, -2, -1), keepdim=True)   # [B, C, 1, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)     # [B, 1, d, h, w]
        cam = F.relu(cam)

        cam_resized = self._resize_like_input(cam, x)
      
      
        return cam_resized


class ScoreCAM3D(BaseCAM3D):
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module, device: torch.device):
        super().__init__(model, target_module, device)
        self.activation: Optional[torch.Tensor] = None
        self.hook = self.target_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        self.activation = out.detach()

    def remove_hooks(self):
        if self.hook is not None:
            self.hook.remove()

    @torch.no_grad()
    def compute(self, x: torch.Tensor, class_index: int) -> torch.Tensor:
        assert x.shape[0] == 1, "ScoreCAM3D expects batch size 1"
        _ = self.model(x)
        if self.activation is None:
            raise RuntimeError("Target layer did not produce activation")

        score_map = torch.zeros_like(x[:, :1, :, :, :], device=x.device)
        num_channels = self.activation.shape[1]

        for c in range(num_channels):
            saliency_map = F.interpolate(
                self.activation[:, c:c+1, :, :, :],
                size=x.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
            if saliency_map.max() == saliency_map.min():
                continue

            norm_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            logits = self.model(x * norm_map)
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)[:, 0]  # [B]
                weight = probs.mean()
            else:
                probs = torch.softmax(logits, dim=1)  # [B, C]
                weight = probs[:, class_index].mean()

            score_map += weight * saliency_map

        score_map = F.relu(score_map)
        
        
        return score_map


# ------------------------- Unified Pipeline -------------------------

def run_cam_pipeline(
    method: str,
    normalization_mode: str,
    config: Dict,
    device: torch.device,
    logger: logging.Logger
) -> None:
    path_root = Path(os.environ.get("root_path"))
    cfg = config.get("cam", config.get("gradcam", {}))

    chkpt_folder = cfg.get("chkpt_folder", "")
    output_dir = path_root / Path(cfg.get("output_dir", "")) /Path(cfg.get("method", ""))/Path(cfg.get("normalization_mode", "")) / Path(chkpt_folder).name
    output_dir.mkdir(parents=True, exist_ok=True)

    list_cases: List[str] = cfg.get("list_cases", None)
    target_class: Optional[int] = cfg.get("target_class", None)
    alpha: float = cfg.get("alpha", 0.35)
    patch_size: List[int] = cfg.get("patch_size", [64, 256, 256])
    patch_overlap: List[int] = cfg.get("patch_overlap", [32, 128, 128])
    dataset_name: Optional[str] = cfg.get("dataset", None)
    use_labels: bool = cfg.get("use_labels", True)

    if not list_cases:
        raise ValueError("Please provide 'list_cases' in config under 'cam' (or legacy 'gradcam').")

    ds_test = ColonCancer(
        dataset_name=dataset_name,
        patch_size=patch_size,
        split='test',
        return_full_image=True,
        use_labels=use_labels
    )

    model = get_model(config)
    model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    target_layer = find_last_conv_layer4(model)
    if method.lower() == "gradcam":
        extractor: BaseCAM3D = GradCAM3D(model, target_layer, device)
    elif method.lower() == "scorecam":
        extractor = ScoreCAM3D(model, target_layer, device)
    else:
        raise ValueError("Unsupported method. Use 'gradcam' or 'scorecam'.")

    all_cams: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]]] = {}
    global_min = float('inf')
    global_max = float('-inf')

    # Compute CAMs
    for uid in tqdm(list_cases, desc=f"Computing {method}"):
        sample = ds_test.get_item_by_uid(uid)
        image, label, _ = sample["source"], sample["label"], sample["target"]

        image = pad_to_shape(image.to(device), tuple(patch_size))
        label = pad_to_shape(label.to(device), tuple(patch_size))

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

            with torch.no_grad():
                logits = model(patch_img)
                if model.out_ch == 1:
                    probs = torch.sigmoid(logits)
                    pred = int((probs > 0.5).item())
                    prob = float(probs.item())
                else:
                    probs = torch.softmax(logits, dim=1)
                    pred = int(probs.argmax(dim=1).item())
                    prob = float(probs[0, pred].item())

            class_index = target_class if target_class is not None else pred
            cam = extractor.compute(patch_img, class_index)

            all_cams[uid].append((cam.detach().cpu(), patch_img.detach().cpu(), patch_lbl.detach().cpu(), patch_idx, pred, prob))
            global_min = min(global_min, float(cam.min().item()))
            global_max = max(global_max, float(cam.max().item()))


    # Visualization and saving
    for uid, cam_list in tqdm(all_cams.items(), desc=f"Saving {method} outputs"):
        out_case_dir = output_dir / f"{uid}"

        if normalization_mode == "uid" and len(cam_list) > 0:
            uid_min = min(float(cam.min().item()) for cam, *_ in cam_list)
            uid_max = max(float(cam.max().item()) for cam, *_ in cam_list)

        for cam, vol, lbl, patch_idx, pred, prob in cam_list:
            if normalization_mode == "patch":
                cmin, cmax = float(cam.min().item()), float(cam.max().item())
            elif normalization_mode == "uid":
                cmin, cmax = uid_min, uid_max
            else:  # global
                cmin, cmax = global_min, global_max

            cam_nii = out_case_dir / f"patch{patch_idx:04d}_cam_pred{pred}_p{prob:.3f}.nii.gz"
            save_cam_nifti(cam, cam_nii)

            out_png = out_case_dir / f"patch{patch_idx:04d}_overlay_pred{pred}_p{prob:.3f}.png"
            overlay_and_save_slices(vol, cam, cmin, cmax, lbl, out_png, alpha=alpha)
            logger.info(f"uid={uid} patch={patch_idx} pred={pred} prob={prob:.3f} saved={cam_nii.name},{out_png.name}")

    extractor.remove_hooks()
    print(f"Saved {method} outputs under: {output_dir}")


def main():
    config_file = Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Backward-compatible: read from 'cam' first, then 'gradcam'
    cfg = config.get("cam")

    method = cfg.get("method", "gradcam")               # 'gradcam' | 'scorecam'
    normalization_mode = cfg.get("normalization_mode", "global")  # 'global' | 'uid' | 'patch'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    logger = logging.getLogger(method.lower())
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())

    path_root = Path(os.environ.get("root_path"))
    chkpt_folder = cfg.get("chkpt_folder", "")
    output_dir = path_root / Path(cfg.get("output_dir", "")) /Path(cfg.get("method", ""))/Path(cfg.get("normalization_mode", "")) / Path(chkpt_folder).name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'{method.lower()}_log.txt'
    logger.addHandler(logging.FileHandler(log_file, mode='a'))

    run_cam_pipeline(method, normalization_mode, config, device, logger)


if __name__ == "__main__":
    main()