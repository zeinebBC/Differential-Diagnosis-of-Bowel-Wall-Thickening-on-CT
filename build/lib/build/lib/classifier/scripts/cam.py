# cam_pipeline_refactored.py

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
from classifier.data import basedataset
from classifier.data.utils.functions_utils import pad_to_shape

# ------------------------- Utilities -------------------------


def get_model(cfg: Dict) -> ResNet:
    model_cfg = cfg["model"]
    if model_cfg["type"] == "ResNet":
        return ResNet(
            in_ch=model_cfg["in_ch"],
            out_ch=model_cfg["out_ch"],
            model=model_cfg["resnet_model"],
            kwargs_resnet=model_cfg["kwargs_resnet"],
        )
    raise ValueError(f"Unknown model type: {model_cfg['type']}")


def find_last_conv_layer4(model: ResNet) -> torch.nn.Module:
    return model.model.layer4[-1].conv2


def save_cam_nifti(cam: torch.Tensor, out_path: Path) -> None:
    cam_data = cam.squeeze(0) if cam.ndim == 5 else cam
    tio.ScalarImage(tensor=cam_data).save(str(out_path))


def overlay_and_save_slices(
    volume: torch.Tensor,
    cam: torch.Tensor,
    min_v: float,
    max_v: float,
    label: Optional[torch.Tensor],
    out_png: Path,
    alpha: float,
    num_slices: int,
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
            ax.imshow(
                np.ma.masked_where(lbl[z, :, :] == 0, lbl[z, :, :]),
                cmap="autumn",
                alpha=0.35,
            )
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


# ------------------------- CAM Classes -------------------------


class BaseCAM3D:
    def __init__(
        self,
        model: torch.nn.Module,
        target_module: torch.nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.target_module = target_module
        self.device = device

    @torch.no_grad()
    def _resize_like_input(
        self, cam: torch.Tensor, input_3d: torch.Tensor
    ) -> torch.Tensor:
        return F.interpolate(
            cam, size=input_3d.shape[-3:], mode="trilinear", align_corners=False
        )

    def compute(self, x: torch.Tensor, class_index: int) -> torch.Tensor:
        raise NotImplementedError

    def remove_hooks(self) -> None:
        pass


class GradCAM3D(BaseCAM3D):
    def __init__(
        self,
        model: torch.nn.Module,
        target_module: torch.nn.Module,
        device: torch.device,
    ):
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
        logits = self.model(x)
        is_binary = logits.shape[1] == 1
        score = logits[:, 0].sum() if is_binary else logits[:, class_index].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(-3, -2, -1), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._resize_like_input(cam, x)


class ScoreCAM3D(BaseCAM3D):
    def __init__(
        self,
        model: torch.nn.Module,
        target_module: torch.nn.Module,
        device: torch.device,
    ):
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
        assert x.shape[0] == 1
        _ = self.model(x)
        if self.activation is None:
            raise RuntimeError("Target layer did not produce activation")

        score_map = torch.zeros_like(x[:, :1, :, :, :], device=x.device)
        num_channels = self.activation.shape[1]

        for c in range(num_channels):
            saliency_map = F.interpolate(
                self.activation[:, c : c + 1, :, :, :],
                size=x.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
            if saliency_map.max() == saliency_map.min():
                continue

            norm_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )
            logits = self.model(x * norm_map)
            weight = (
                torch.sigmoid(logits)[:, 0].mean()
                if logits.shape[1] == 1
                else torch.softmax(logits, dim=1)[:, class_index].mean()
            )
            score_map += weight * saliency_map

        return F.relu(score_map)


# ------------------------- Pipeline -------------------------


def run_cam_pipeline(
    method: str,
    normalization_mode: str,
    cfg: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    path_root = Path(os.environ.get("root_path"))
    cam_cfg = cfg["cam"]  # now no .get(), all keys must exist

    chkpt_folder = Path(cam_cfg["chkpt_folder"])
    output_dir = (
        path_root
        / cam_cfg["output_dir"]
        / cam_cfg["method"]
        / cam_cfg["normalization_mode"]
        / chkpt_folder.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    list_cases: List[str] = cam_cfg["list_cases"]
    target_class: int = cam_cfg["target_class"]
    alpha: float = cam_cfg["alpha"]
    patch_size: List[int] = cam_cfg["patch_size"]
    patch_overlap: List[int] = cam_cfg["patch_overlap"]
    dataset_name: str = cam_cfg["dataset"]
    use_labels: bool = cam_cfg["use_labels"]

    ds_test = basedataset(
        dataset_name=dataset_name,
        patch_size=patch_size,
        split="test",
        return_full_image=True,
        use_labels=use_labels,
    )

    model = get_model(cfg)
    model = model.load_best_checkpoint(chkpt_folder)
    model.to(device)
    model.eval()

    target_layer = find_last_conv_layer4(model)
    extractor: BaseCAM3D = (
        GradCAM3D(model, target_layer, device)
        if method.lower() == "gradcam"
        else ScoreCAM3D(model, target_layer, device)
    )

    all_cams: Dict[
        str, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]]
    ] = {}
    global_min, global_max = float("inf"), float("-inf")

    for uid in tqdm(list_cases, desc=f"Computing {method}"):
        sample = ds_test.get_item_by_uid(uid)
        image, label, _ = sample["source"], sample["label"], sample["target"]

        image = pad_to_shape(image.to(device), tuple(patch_size))
        label = pad_to_shape(label.to(device), tuple(patch_size))

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image[0].detach().cpu()),
            label=tio.LabelMap(tensor=label[0].detach().cpu()),
        )
        sampler = tio.inference.GridSampler(
            subject, patch_size=patch_size, patch_overlap=patch_overlap
        )

        all_cams[uid] = []
        out_case_dir = output_dir / uid
        out_case_dir.mkdir(parents=True, exist_ok=True)

        for patch_idx, patch in enumerate(sampler):
            patch_img = patch["image"][tio.DATA].unsqueeze(0).to(device)
            patch_lbl = patch["label"][tio.DATA].unsqueeze(0).to(device)

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
            all_cams[uid].append(
                (
                    cam.detach().cpu(),
                    patch_img.detach().cpu(),
                    patch_lbl.detach().cpu(),
                    patch_idx,
                    pred,
                    prob,
                )
            )
            global_min = min(global_min, float(cam.min().item()))
            global_max = max(global_max, float(cam.max().item()))

    # Visualization
    for uid, cam_list in tqdm(all_cams.items(), desc=f"Saving {method} outputs"):
        out_case_dir = output_dir / uid
        uid_min = min(float(cam.min().item()) for cam, *_ in cam_list)
        uid_max = max(float(cam.max().item()) for cam, *_ in cam_list)

        for cam, vol, lbl, patch_idx, pred, prob in cam_list:
            if normalization_mode == "patch":
                cmin, cmax = float(cam.min().item()), float(cam.max().item())
            elif normalization_mode == "uid":
                cmin, cmax = uid_min, uid_max
            else:
                cmin, cmax = global_min, global_max

            cam_nii = (
                out_case_dir / f"patch{patch_idx:04d}_cam_pred{pred}_p{prob:.3f}.nii.gz"
            )
            save_cam_nifti(cam, cam_nii)

            out_png = (
                out_case_dir
                / f"patch{patch_idx:04d}_overlay_pred{pred}_p{prob:.3f}.png"
            )
            overlay_and_save_slices(
                vol,
                cam,
                cmin,
                cmax,
                lbl,
                out_png,
                alpha=alpha,
                num_slices=cam_cfg["num_slices"],
            )
            logger.info(
                f"uid={uid} patch={patch_idx} pred={pred} prob={prob:.3f} saved={cam_nii.name},{out_png.name}"
            )

    extractor.remove_hooks()
    logger.info(f"Saved {method} outputs under: {output_dir}")


def main():
    config_file = Path(__file__).resolve().parent.parent / "run_config.json"
    with open(config_file, "r") as f:
        cfg = json.load(f)

    method = cfg["cam"]["method"]
    normalization_mode = cfg["cam"]["normalization_mode"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    logger = logging.getLogger(method.lower())
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler())

    run_cam_pipeline(method, normalization_mode, cfg, device, logger)


if __name__ == "__main__":
    main()
