import SimpleITK as sitk
import numpy as np

def read_nii(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

def to_2d_slice(arr: np.ndarray):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    raise ValueError(f"Unsupported shape: {arr.shape}")

def norm_to_uint8(x):
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)
