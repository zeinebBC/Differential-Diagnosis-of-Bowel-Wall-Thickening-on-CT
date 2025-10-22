import torchvision.transforms.functional as fT
import torch.nn.functional as F
import random
import numpy as np
import nibabel as nib
import torch

def pad_to_shape(vol, target_shape, constant_values=0):
    pad_width = []
    current_shape = vol.shape[-3:]
    for i in range(3):
        total_pad = max(0, target_shape[i] - current_shape[i])
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    if isinstance(vol, torch.Tensor):
        pad_tuple = tuple([elem for tup in reversed(pad_width) for elem in tup])
        return F.pad(vol, pad_tuple, mode="constant", value=constant_values)
    else:
        leading_dims = len(vol.shape) - 3
        pad_width_full = [(0, 0)] * leading_dims + pad_width
        return np.pad(vol, pad_width_full, mode="constant", constant_values=constant_values)
    
def pad_batch_with_channel(vol, target_shape, constant_values=0):
    padded_vols = []
    for vol in vol:
        padded_vol = pad_to_shape(vol, target_shape, constant_values)
        padded_vols.append(padded_vol)
    return torch.stack(padded_vols)

def load_volume(path):
            if path.suffix == ".npz":
                return np.load(path)["image"]
            elif path.suffix in [".nii", ".nii.gz", ".gz"]:
                return nib.load(str(path)).get_fdata()
            else:
                raise ValueError(f"Unsupported image format: {path.suffix}")
            


def transform_image_and_label(image, label):
    # Horizontal flip
    if random.random() < 0.4:
        image = fT.hflip(image)
        label = fT.hflip(label)

    # Vertical flip
    if random.random() < 0.4:
        image = fT.vflip(image)
        label = fT.vflip(label)

    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-10, 10)
        image = fT.rotate(image, angle)
        label = fT.rotate(label, angle, interpolation=F.InterpolationMode.NEAREST)

    return image, label







