import numpy as np
from scipy.ndimage import binary_fill_holes
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
import os
import sys

from pathlib import Path
from typing import  Tuple, Sequence

import SimpleITK as sitk

def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox
def voxels_from_mm( spacing_xyz: Tuple[float, float, float],margin_mm: Tuple[float, float, float]=(10,10,10)) -> Tuple[int, int, int]:
    return tuple(int(round(mm / sp)) for mm, sp in zip(margin_mm, spacing_xyz))

def get_bbox_from_mask_with_margin(mask: np.ndarray, margin_voxels: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    """
    Compute tight bounding box around nonzero mask, expand by margin_voxels.
    Returns 3D slices usable to crop arrays: arr[xs, ys, zs].
    """
    assert mask.ndim == 3
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        # empty mask -> whole volume slices
        return [[0, int(mask.shape[0])], [0, int(mask.shape[1])], [0, int(mask.shape[2])]]
    minz = coords.min(axis=0)
    maxz = coords.max(axis=0)
    x0, y0, z0 = minz
    x1, y1, z1 = maxz
    x0 = max(0, x0 - margin_voxels[0])
    y0 = max(0, y0 - margin_voxels[1])
    z0 = max(0, z0 - margin_voxels[2])
    x1 = min(mask.shape[0] - 1, x1 + margin_voxels[0])
    y1 = min(mask.shape[1] - 1, y1 + margin_voxels[1])
    z1 = min(mask.shape[2] - 1, z1 + margin_voxels[2])
    return [[int(x0), int(x1 + 1)], [int(y0), int(y1 + 1)], [int(z0), int(z1 + 1)]]
def crop_to_bbox_no_channels(image, bbox: Sequence[Sequence[int]]):
    """
    Crops image to bounding box (in spatial dimensions)

    Args:
        image (arraylike): 2d or 3d array
        bbox (Sequence[Sequence[int]]): bounding box coordinated in an interleaved fashion
            (e.g. (x1, x2), (y1, y2), (z1, z2))

    Returns:
        arraylike: cropped array
    """
    resizer = tuple([slice(_dim[0], _dim[1]) for _dim in bbox])
    return image[resizer]


def crop_to_bbox(data: np.ndarray, bbox: Sequence[Sequence[int]]):
    """
    Crops image to bounding box (performed per channel)

    Args:
        data (np.ndarray): 3d or 4d array [C, X, Y, (Z)]
        bbox (Sequence[Sequence[int]]): bounding box coordinated in an interleaved fashion
            (e.g. (x1, x2), (y1, y2), (z1, z2))

    Returns:
        np.ndarray: cropped array
    """
    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox_no_channels(data[c], bbox)
        cropped_data.append(cropped)
    data = np.stack(cropped_data)
    return data

def crop_to_label(data, seg, spacing, margin_min=20):
    """
    Crop image & segmentation to a region of interest defined by the label (segmentation mask).

    Args:
        data (np.ndarray): 4D image [C, D, H, W]
        seg (np.ndarray): 3D segmentation [D, H, W]
        spacing (Tuple[float]): voxel spacing in mm
        margin_min (float): margin (in mm) added around the labeled region

    Returns:
        cropped_data (np.ndarray): cropped image
        cropped_seg (np.ndarray): cropped segmentation
        bbox (List[Tuple[int, int]]): voxel bounding box
    """
    margin_vox = voxels_from_mm(spacing, (margin_min, margin_min, margin_min))
    bbox = get_bbox_from_mask_with_margin(seg, margin_vox)

    data_cropped = crop_to_bbox(data, bbox)
    seg_cropped = crop_to_bbox_no_channels(seg, bbox)

    return data_cropped, seg_cropped, bbox

def crop_to_colon(data, seg, case_id, spacing, margin_min=20, margin_step=1):
    """
    Crop data to colon region

    Args:
        data (np.ndarray): data to crop
        seg (np.ndarray): segmentation
        nonzero_label (int): nonzero label is written into segmentation map
            where only background was found

    Returns:
        np.ndarray: cropped data
        np.ndarray: cropped and filled (with nonzero_label) segmentation
        List[Tuple[int]]: bounding box of nonzero region
    """
    
    try:
        masks_root = os.environ["auto_seg"]
    except KeyError:
        print("Error: Environment variable auto_seg is not set.", file=sys.stderr)
        sys.exit(1)

   
    mask_path = Path(masks_root) / case_id
    if not mask_path.exists():
        print(f"No mask found for {case_id} in {masks_root}")
        bbox = [[0, data.shape[1] - 1],[0, data.shape[2] - 1],[0, data.shape[3] - 1]]
        return data, seg, bbox

    
    mask_colon = sitk.ReadImage(str(mask_path))
    mask_colon = sitk.GetArrayFromImage(mask_colon)[None].astype(np.float32)[0]
    """
    if seg is None:
        

        margin_vox = voxels_from_mm(spacing, margin_mm=(margin_min, margin_min, margin_min))
        bbox = get_bbox_from_mask_with_margin(mask_colon, margin_vox)

        data_cropped = crop_to_bbox(data, bbox)
        return data_cropped,seg, bbox

    """

    
    if seg is None:
        """
        # No segmentation: return original data and full bounding box in correct format
        bbox = [
            [0, data.shape[1]],  # X dimension
            [0, data.shape[2]],  # Y dimension
            [0, data.shape[3]]   # Z dimension
        ]
        return data, seg, bbox
        """
        margin_vox = voxels_from_mm(spacing, margin_mm=(margin_min, margin_min, margin_min))
        bbox = get_bbox_from_mask_with_margin(mask_colon, margin_vox)

        data_cropped = crop_to_bbox(data, bbox)
        return data_cropped,seg, bbox

    
    else: 
        original_sum = np.sum(seg)
        margin_mm = margin_min
        while True:
            margin_vox = voxels_from_mm( spacing, margin_mm=(margin_mm, margin_mm, margin_mm))
            bbox = get_bbox_from_mask_with_margin(mask_colon, margin_vox)

            data_cropped = crop_to_bbox(data, bbox)
            seg_cropped = crop_to_bbox(seg, bbox)

            if np.sum(seg_cropped ) == original_sum:
                    break  # safe margin found
            else:
                #print(f"  ALERT: Cropping removed part of GT for {case_id} with margin {margin_mm} mm")
                margin_mm += margin_step
            

       
        return data_cropped, seg_cropped, bbox
