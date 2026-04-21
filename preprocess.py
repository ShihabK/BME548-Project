"""
preprocess.py

Data loading, color space transforms, stain perturbation simulation,
and PyTorch Dataset/DataLoader construction for the AML-Cytomorphology project.

Pipeline overview:
    Raw TIFF (sRGB)
    * ToTensor()  =>  [0, 1] float tensor
    * RGBToLab    (differentiable physical layer, no learned params)
    * correction module  (defined in models.py)
    * LabToRGBa
    * ImageNet normalisation  (for ConvNeXt-Tiny pretrained weights)

Usage:
    from preprocess import build_dataloaders, RGBToLab, LabToRGB
    train_loader, val_loader, test_loader, class_names, class_to_idx, abbrev_map = (
        build_dataloaders()
    )
"""

import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split


# 0.  Constants

DATA_ROOT = "/content/drive/MyDrive/BME548_Final_Project"
IMAGE_FOLDER = "PKG - AML-Cytomorphology_LMU"
IMAGE_SIZE = 224
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42

VALID_CLASSES = [
    "BAS", "EBO", "EOS", "KSC", "LYA", "LYT",
    "MMZ", "MOB", "MON", "MYB", "MYO", "NGB",
    "NGS", "PMB", "PMO",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 1.  Parse annotation and abbreviation files

def parse_abbreviations(filepath):
    """
    Parse abbreviations.txt into a dict mapping abbreviation.

    Returns
    -------
    dict[str, str]  e.g. {"BAS": "Basophil", "EBO": "Erythroblast", ...}
    """
    abbrev_map = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            match = re.match(r"^([A-Z]{2,3})\s+(.+)$", line)
            if match:
                abbrev_map[match.group(1)] = match.group(2)
    return abbrev_map


def parse_annotations(filepath):
    """
    Parse annotations.dat into a filtered DataFrame.

    Columns in the file (no header, whitespace-separated):
        col 0 : relative path  e.g. BAS/BAS_0001.tiff
        col 1 : original label
        col 2 : first re-annotator  (literal "nan" if not re-annotated)
        col 3 : second re-annotator (literal "nan" if not re-annotated)

    Filtering applied:
        - Rows whose original label is not in VALID_CLASSES are dropped.
        - Rows where either re-annotator marked the image UNC are dropped.

    Returns
    -------
    pd.DataFrame with columns [filepath, label, reanno_1, reanno_2]
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["filepath", "label", "reanno_1", "reanno_2"],
    )
    df.replace("nan", np.nan, inplace=True)
    df = df[df["label"].isin(VALID_CLASSES)].copy()
    unc_mask = (df["reanno_1"] == "UNC") | (df["reanno_2"] == "UNC")
    df = df[~unc_mask].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def resolve_image_paths(df, data_root, image_folder):
    """
    Add an abs_path column by prepending the full base path to each filepath.
    """
    base = os.path.join(data_root, image_folder)
    df = df.copy()
    df["abs_path"] = df["filepath"].apply(
        lambda p: os.path.join(base, p))
    return df


# 2.  Train / val / test split  (stratified by class)

def split_dataframe(df, ratios=SPLIT_RATIOS, seed=SEED):
    """
    Stratified split into train, val, and test sets due to rare classes.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1."
    train_r, val_r, test_r = ratios

    train_val_df, test_df = train_test_split(
        df, test_size=test_r, stratify=df["label"], random_state=seed,
    )
    relative_val = val_r / (train_r + val_r)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val,
        stratify=train_val_df["label"], random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# 3.  Differentiable sRGB to CIE L*a*b* transforms and vice versa

class RGBToLab(nn.Module):
    """
    Differentiable sRGB to CIE L*a*b* transform.

    Input  : (B, 3, H, W) float tensor, values in [0, 1]  <-- raw pixels
    Output : (B, 3, H, W) float tensor
                channel 0 -> L*  in [0, 100]
                channel 1 -> a*  approx [-128, 127]
                channel 2 -> b*  approx [-128, 127]
    """

    _M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float32)

    _WHITE = torch.tensor(
        [0.95047, 1.00000, 1.08883], dtype=torch.float32)

    def forward(self, rgb):
        # 1. Gamma inversion
        rgb_lin = torch.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4,
        )

        # 2. Linear RGB to XYZ (D65)
        M = self._M.to(rgb.device)
        B, _, H, W = rgb_lin.shape
        xyz = torch.einsum(
            "ij,bjk->bik", M, rgb_lin.view(B, 3, -1)).view(B, 3, H, W)

        # 3. Normalise by D65 white point
        white = self._WHITE.to(rgb.device).view(1, 3, 1, 1)
        xyz_n = xyz / white

        # 4. CIE cube root companding
        delta = 6.0 / 29.0
        xyz_f = torch.where(
            xyz_n > delta ** 3,
            xyz_n.clamp(min=1e-8) ** (1.0 / 3.0),
            xyz_n / (3 * delta ** 2) + 4.0 / 29.0,
        )

        # 5. L*, a*, b*
        L = 116.0 * xyz_f[:, 1:2] - 16.0
        a = 500.0 * (xyz_f[:, 0:1] - xyz_f[:, 1:2])
        b = 200.0 * (xyz_f[:, 1:2] - xyz_f[:, 2:3])

        return torch.cat([L, a, b], dim=1)


class LabToRGB(nn.Module):
    """
    Differentiable CIE L*a*b* to sRGB transform (inverse of RGBToLab).

    Input  : (B, 3, H, W) Lab tensor
    Output : (B, 3, H, W) sRGB tensor clamped to [0, 1]
    """

    _M_inv = torch.tensor([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=torch.float32)

    _WHITE = torch.tensor(
        [0.95047, 1.00000, 1.08883], dtype=torch.float32)

    def forward(self, lab):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

        # 1. Lab to XYZ
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        delta = 6.0 / 29.0

        def f_inv(t):
            return torch.where(t > delta, t ** 3, 3 *
                               delta ** 2 * (t - 4.0 / 29.0))

        white = self._WHITE.to(lab.device).view(1, 3, 1, 1)
        xyz = torch.cat(
            [f_inv(fx), f_inv(fy), f_inv(fz)], dim=1) * white

        # 2. XYZ to linear RGB
        M_inv = self._M_inv.to(lab.device)
        B, _, H, W = xyz.shape
        rgb_lin = torch.einsum(
            "ij,bjk->bik", M_inv, xyz.view(B, 3, -1)
        ).view(B, 3, H, W)

        # 3. Gamma encoding
        rgb = torch.where(
            rgb_lin <= 0.0031308,
            12.92 * rgb_lin,
            1.055 * rgb_lin.clamp(min=1e-8) ** (1.0 / 2.4) - 0.055,
        )
        return rgb.clamp(0.0, 1.0)


# 4.  Stain perturbation  (HSV-space)

def perturb_staining(image_np, sigma_s=0.05,
                     sigma_c=0.05, sigma_w=0.03, rng=None):
    """
    Simulate staining variability via small random shifts in HSV space.

    HSV maps cleanly onto the three physical sources of staining variability:
      - Hue        -> stain spectrum shifts (sigma_s)
                      models reagent batch and pH variability
      - Saturation -> stain concentration shifts (sigma_c)
                      models incubation time and wash protocol variability
      - Value      -> illuminant / white balance shifts (sigma_w)
                      models microscope light source differences

    The white balance perturbation is applied in linear RGB space (after gamma
    inversion) before the HSV shift, since real illuminant changes occur before
    the camera's gamma encoding step.

    Parameters
    ----------
    image_np : np.ndarray (H, W, 3) uint8
    sigma_s  : float — std of Gaussian hue shift       (stain spectrum)
    sigma_c  : float — std of multiplicative sat shift  (concentration)
    sigma_w  : float — std of multiplicative val shift  (white balance)
    rng      : np.random.Generator or None

    Returns
    -------
    np.ndarray (H, W, 3) uint8
    """
    if rng is None:
        rng = np.random.default_rng()

    img_f = image_np.astype(np.float32) / 255.0

    # White balance in linear RGB
    img_lin = np.where(
        img_f <= 0.04045, img_f / 12.92,
        ((img_f + 0.055) / 1.055) ** 2.4,
    )
    img_lin = np.clip(
        img_lin * (1.0 + rng.normal(0, sigma_w, (3,)))[None, None, :],
        0.0, 1.0,
    )
    img_f = np.clip(
        np.where(img_lin <= 0.0031308, 12.92 * img_lin,
                 1.055 * img_lin ** (1.0 / 2.4) - 0.055),
        0.0, 1.0,
    )

    # RGB to HSV
    r, g, b = img_f[..., 0], img_f[..., 1], img_f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    chroma = maxc - minc  # (H, W)

    # Saturation
    s = np.zeros_like(maxc)
    np.divide(chroma, maxc, out=s, where=maxc > 1e-6)

    # Hue
    safe = np.where(chroma > 1e-6, chroma, 1.0)  # denominator guard
    h_r = ((g - b) / safe) % 6.0
    h_g = (b - r) / safe + 2.0
    h_b = (r - g) / safe + 4.0

    h = np.where(maxc == r, h_r,
                 np.where(maxc == g, h_g, h_b))
    h = np.where(chroma > 1e-6, h / 6.0, 0.0)  # normalise to [0, 1]

    # Hue and saturation perturbations
    h = (h + rng.normal(0, sigma_s)) % 1.0
    s = np.clip(s * (1.0 + rng.normal(0, sigma_c)), 0.0, 1.0)

    # HSV to RGB
    h6 = h * 6.0
    hi = np.floor(h6).astype(np.int32) % 6   # sector index 0-5  (H, W)
    f = h6 - np.floor(h6)                   # fractional part

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # Stack all six (r, g, b) outcomes: shape (6, H, W, 3)
    sector_rgb = np.stack([
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ], axis=0)   # (6, H, W, 3)

    rgb_out = sector_rgb[hi, np.arange(hi.shape[0])[:, None],
                         np.arange(hi.shape[1])[None, :], :]

    return np.clip(rgb_out * 255.0, 0, 255).astype(np.uint8)


# 5.  PyTorch Dataset

class AMLDataset(Dataset):
    """
    PyTorch Dataset for the AML-Cytomorphology-LMU dataset.

    Returns (image_tensor, label_index) where image_tensor is in [0, 1].

    When augment=True (training), returns (perturbed_tensor, clean_tensor, label)
    so the training loop can compute the stain consistency loss between the
    perturbed and clean versions after correction.

    When augment=False (val/test), returns (image_tensor, label) as usual.

    Parameters
    ----------
    df           : pd.DataFrame with columns [abs_path, label]
    class_to_idx : dict[str, int]
    augment      : bool — apply stain perturbation and geometric augmentation
    sigma_s/c/w  : float — perturbation magnitudes
    """

    def __init__(
        self,
        df,
        class_to_idx,
        augment=False,
        sigma_s=0.05,
        sigma_c=0.05,
        sigma_w=0.03,
    ):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.augment = augment
        self.sigma_s = sigma_s
        self.sigma_c = sigma_c
        self.sigma_w = sigma_w
        self.rng = np.random.default_rng(SEED)
        self.eval_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _apply_same_augmentation(self, pil_clean, pil_perturbed):
        """
        Apply the same random geometric transform to both images.

        Uses torchvision.transforms.functional so we can sample the random
        parameters once and apply them identically to both the clean and
        perturbed versions.  This ensures pixel-level alignment for the
        consistency loss.
        """
        # Resize
        pil_clean = TF.resize(pil_clean, (IMAGE_SIZE, IMAGE_SIZE))
        pil_perturbed = TF.resize(
            pil_perturbed, (IMAGE_SIZE, IMAGE_SIZE))

        # Random horizontal flip
        if random.random() > 0.5:
            pil_clean = TF.hflip(pil_clean)
            pil_perturbed = TF.hflip(pil_perturbed)

        # Random vertical flip
        if random.random() > 0.5:
            pil_clean = TF.vflip(pil_clean)
            pil_perturbed = TF.vflip(pil_perturbed)

        # Random rotation
        angle = random.uniform(-180, 180)
        pil_clean = TF.rotate(pil_clean, angle)
        pil_perturbed = TF.rotate(pil_perturbed, angle)

        return TF.to_tensor(pil_clean), TF.to_tensor(pil_perturbed)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["abs_path"]).convert("RGB")
        img_np = np.array(img)
        label = self.class_to_idx[row["label"]]

        if self.augment:
            # Create stain-perturbed version
            perturbed_np = perturb_staining(
                img_np,
                sigma_s=self.sigma_s,
                sigma_c=self.sigma_c,
                sigma_w=self.sigma_w,
                rng=self.rng,
            )

            # Apply identical geometric augmentation to both
            clean_tensor, perturbed_tensor = self._apply_same_augmentation(
                Image.fromarray(img_np),
                Image.fromarray(perturbed_np),
            )

            # Return (perturbed, clean, label)
            # perturbed goes through the full pipeline for classification; both go through the
            # correction module for the consistency loss
            return perturbed_tensor, clean_tensor, label
        else:
            img_tensor = self.eval_transform(Image.fromarray(img_np))
            return img_tensor, label


# 6.  Extract dataset from zip to local SSD (Google Colab specific)
# Note from Shihab:
# I added this because loading dataset from drive is really slow.
# Multiple workers are not safe in google drive.
# Fastest option is to extract the dataset to local SSD and load from
# there with num_workers > 0.

def extract_dataset_to_local(
    zip_path="/content/drive/MyDrive/BME548_Final_Project/aml_dataset.zip",
    local_root="/tmp/aml_dataset",
):
    """
    Extract the dataset zip from Drive to Colab's local /tmp SSD.

    Returns the full path to the dataset folder (local_root/dataset),
    which is what build_dataloaders() expects as data_root.

    Parameters
    ----------
    zip_path   : str — path to the zip file on Drive
    local_root : str — extraction root on local SSD

    Returns
    -------
    str — path to the dataset folder inside local_root
    """
    import zipfile
    import time

    dataset_dir = os.path.join(local_root, "dataset")

    if os.path.isdir(dataset_dir):
        n = sum(len(f) for _, _, f in os.walk(dataset_dir))
        print(
            f"Dataset already extracted at {dataset_dir}  ({n} files). Skipping.")
        return dataset_dir

    assert os.path.isfile(zip_path), (
        f"Zip not found at {zip_path}."
    )

    os.makedirs(local_root, exist_ok=True)
    print(f"Extracting {zip_path} -> {local_root} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(local_root)
    elapsed = time.time() - t0

    n = sum(len(f) for _, _, f in os.walk(dataset_dir))
    print(f"Done. Extracted {n} files in {elapsed:.1f}s")
    return dataset_dir


# 7.  Top-level builder

def build_dataloaders(
    data_root=DATA_ROOT,
    image_folder=IMAGE_FOLDER,
    batch_size=32,
    num_workers=0,
    sigma_s=0.05,
    sigma_c=0.05,
    sigma_w=0.03,
    split_ratios=SPLIT_RATIOS,
    seed=SEED,
):
    # Note from Shihab:
    # Google Drive cannot be shared across DataLoader worker processes, so
    # num_workers must be 0 when image paths point into /content/drive/.
    # To use multiple workers, first extract dataset to local SSD with extract_dataset_to_local(),
    # then pass the returned local path as data_root here.

    # Example:
    #     local_root = extract_dataset_to_local()
    #     train_loader, ... = build_dataloaders(
    #         data_root=local_root, num_workers=4
    #     )
    """
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader  (tensors in [0, 1])
    class_names  : list[str]
    class_to_idx : dict[str, int]
    abbrev_map   : dict[str, str]
    """
    abbrev_map = parse_abbreviations(
        os.path.join(data_root, "abbreviations.txt"))
    df = parse_annotations(os.path.join(data_root, "annotations.dat"))
    df = resolve_image_paths(df, data_root, image_folder)

    class_names = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total images after filtering: {len(df)}\n")
    print("Class distribution:")
    for c in class_names:
        n = int((df["label"] == c).sum())
        print(f"  {c:4s}  {abbrev_map.get(c, c):30s}  {n:5d}")

    train_df, val_df, test_df = split_dataframe(
        df, ratios=split_ratios, seed=seed)
    print(
        f"\nSplit — train: {
            len(train_df)}, val: {
            len(val_df)}, test: {
                len(test_df)}")

    pin = torch.cuda.is_available()

    train_ds = AMLDataset(train_df, class_to_idx, augment=True,
                          sigma_s=sigma_s, sigma_c=sigma_c, sigma_w=sigma_w)
    val_ds = AMLDataset(val_df, class_to_idx, augment=False)
    test_ds = AMLDataset(test_df, class_to_idx, augment=False)

    # persistent_workers=True keeps worker processes alive between
    # batches,
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader, class_names, class_to_idx, abbrev_map
