import torch 
from typing import Tuple, Optional
from typing import Optional, Tuple, List
import torch.nn.functional as F 
import nibabel as nib
from pathlib import Path
from data import ColonCancer
import numpy as np
import matplotlib.pyplot as plt
from models import ResNet
import torchio as tio 
class GradCAM3D:
    """
    Minimal Grad-CAM for 3D ResNet-like models.

    target_module should be the last convolutional block whose output feature map
    feeds the global pooling (typically layer4 or its last block).
    """
    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module, device: torch.device):
        self.model = model
        self.target_module = target_module
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.fwd_handle = None
        self.bwd_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out is a tuple with same shape as out
            self.gradients = grad_out[0].detach()

        self.fwd_handle = self.target_module.register_forward_hook(fwd_hook)
        self.bwd_handle = self.target_module.register_full_backward_hook(bwd_hook)

    def remove_hooks(self):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()

    @torch.no_grad()
    def _resize_like_input(self, cam: torch.Tensor, input_3d: torch.Tensor) -> torch.Tensor:
        # cam: [B, 1, d, h, w], input_3d: [B, C, D, H, W]
        target_size = input_3d.shape[-3:]
        return F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)

    def __call__(self, input_3d: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Returns a CAM normalized to [0, 1], shape [B, 1, D, H, W].
        """
        self.model.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        logits = self.model(input_3d)  # [B, num_classes]
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward on the target class score
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_module selection.")

        # activations: [B, C, d, h, w]
        # gradients:   [B, C, d, h, w]
        weights = self.gradients.mean(dim=(-3, -2, -1), keepdim=True)  # [B, C, 1, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)    # [B, 1, d, h, w]
        cam = F.relu(cam)

        # Normalize per-sample to [0, 1]
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-6)

        cam_resized = self._resize_like_input(cam_norm, input_3d)
        return cam_resized


def find_target_module_for_resnet3d(model: ResNet) -> torch.nn.Module:
    """
    Our ResNet wraps a MONAI ResNetFeatures in nn.Sequential:
      [0] -> ResNetFeatures
      [1] -> GetLast
      [2] -> AdaptiveAvgPool3d
      [3] -> Flatten
      [4] -> Linear

    We hook the last block in layer4 inside ResNetFeatures.
    """
    
    # layer4 is a Sequential of BasicBlock/Bottleneck; hook the last block.
    last_block = model.model.layer4[-1]
    return last_block


def save_cam_nifti(cam: torch.Tensor, out_path: Path):
    """
    Save CAM as NIfTI using TorchIO to preserve basic orientation from the tensor.
    cam: [1, 1, D, H, W] on CPU
    """
   
    cam_data = cam.squeeze(0) if cam.ndim == 5 else cam
    tio.ScalarImage(tensor=cam_data).save(str(out_path))



def overlay_and_save_slices(
    volume: torch.Tensor,
    cam: torch.Tensor,
    label: Optional[torch.Tensor],
    out_png: Path,
    alpha: float = 0.20,
    num_slices: int = 6,
):
    """
    Save a grid of axial slices:
      Row 1: image
      Row 2: image + CAM
      Row 3: image + label (if provided)
    volume: [1, 1, D, H, W] CPU
    cam:    [1, 1, D, H, W] CPU (0..1)
    label:  [1, 1, D, H, W] CPU or None
    """
    vol = volume.squeeze().numpy()
    heat = cam.squeeze().numpy()
    lbl = label.squeeze().numpy() if label is not None else None
    if vol.ndim == 4:
        vol = vol[0,...]
    H,W,D = vol.shape
    z_slices = np.linspace(0, D - 1, num_slices, dtype=int)

    rows = 3 if label is not None else 2
    fig, axes = plt.subplots(rows, num_slices, figsize=(2.5 * num_slices, 2.5 * rows))

    if num_slices == 1:
        axes = np.array([[axes]]) if rows == 1 else np.array([axes])

    # Row 1: image
    for i, z in enumerate(z_slices):
        ax = axes[0, i] if rows > 1 else axes[i]
        ax.imshow(vol[:,:,z], cmap="gray")
        ax.set_title(f"z={z}")
        ax.axis("off")

    # Row 2: image + CAM
    for i, z in enumerate(z_slices):
        ax = axes[1, i] if rows > 1 else axes[i]
        ax.imshow(vol[:,:,z], cmap="gray")
        ax.imshow(heat[:,:,z], cmap="jet", alpha=alpha, vmin=0, vmax=1)
        ax.axis("off")

    # Row 3: image + label
    if label is not None:
        for i, z in enumerate(z_slices):
            ax = axes[2, i]
            ax.imshow(vol[:,:,z], cmap="gray")
            # show label as contour/overlay
            ax.imshow(np.ma.masked_where(lbl[:,:,z] == 0, lbl[:,:,z]), cmap="autumn", alpha=0.35)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def load_case_from_dataset(ds: ColonCancer, uid: int,  in_ch=1) -> Tuple[torch.Tensor, str]:
    match = [item for item in ds.images if item[0] == str(uid)]
    if not match:
        raise ValueError(f"UID '{uid}' not found in dataset.")
    if ds.labels_path is not None:
        label_path = ds.labels_path / f"{uid}.nii.gz"
    else: 
        if ds.split != 'test':
            label_path = f"/data/colon_cancer/Classifier/resampledTr/labels_resampled/{uid}.nii.gz"
        else:
            label_path = f"/data/colon_cancer/Classifier/resampledTs/labels_resampled/{uid}.nii.gz"        
    label = nib.load(label_path).get_fdata()
    uid, img_path, target = match[0]
    # Load image
    if img_path.suffix == ".npz":
        data = np.load(img_path)["image"]
    elif img_path.suffix in [".nii", ".gz", ".nii.gz"]:
        data = nib.load(str(img_path)).get_fdata()
    else:
        raise ValueError(f"Unsupported image format: {img_path.suffix}")
    
    if in_ch == 1:
        data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)   
        label = torch.from_numpy(label).long().unsqueeze(0).unsqueeze(0)
    else:
        data = torch.from_numpy(data).float().unsqueeze(0)               
        label = torch.from_numpy(label).float().unsqueeze(0)            
        data = torch.cat([data, label], dim=0).unsqueeze(0)  
        label = label.unsqueeze(0)

    
    return data, label, torch.tensor(target, dtype=torch.long)
    
