"""
helper.py

Evaluation utilities for the AML stain correction + classification pipeline.

Contents:
    1. compute_metrics        — per-class precision, recall, F1; macro averages
    2. plot_confusion_matrix  — normalised confusion matrix heatmap
    3. robustness_curve       — macro-F1 vs perturbation severity
    4. GradCAM                — Grad-CAM attention maps on ConvNeXt-Tiny
    5. plot_gradcam           — overlay Grad-CAM on original image
    6. plot_lab_distributions — a*/b* chromatic scatter
    before vs after correction

Usage:
    from helpers import (
        compute_metrics, plot_confusion_matrix,
        robustness_curve, GradCAM, plot_gradcam,
        plot_lab_distributions,
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from preprocess import perturb_staining, AMLDataset, IMAGENET_MEAN, IMAGENET_STD
from PIL import Image


def _to_device(batch, device):
    imgs, labels = batch
    return imgs.to(device), labels.to(device)


def denormalise(tensor):
    """
    Reverse ImageNet normalisation to recover [0, 1] pixel values.
    Used before visualising tensors that have passed through the full pipeline.

    Parameters:
    tensor : torch.Tensor (B, 3, H, W) or (3, H, W)
    """
    mean = torch.tensor(
        IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(
        IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


def compute_metrics(model, loader, class_names, device):
    """
    Run inference on a DataLoader and compute per-class and macro metrics.

    Parameters:
    model       : AMLPipeline
    loader      : DataLoader
    class_names : list[str]
    device      : torch.device

    Returns:
    report : dict
        {
          "per_class": pd.DataFrame with columns
                       [class, precision, recall, f1, support],
          "macro_f1" : float,
          "macro_precision": float,
          "macro_recall"   : float,
          "all_preds"      : np.ndarray,
          "all_labels"     : np.ndarray,
        }
    """

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            imgs, labels = _to_device(batch, device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class_df = pd.DataFrame({
        "class": class_names,
        "precision": np.round(precision, 4),
        "recall": np.round(recall, 4),
        "f1": np.round(f1, 4),
        "support": support.astype(int),
    })

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(
        f"\n{
            'Class':<6}  {
            'Precision':>9}  {
                'Recall':>6}  {
                    'F1':>6}  {
                        'Support':>7}")
    print("-" * 46)
    for _, row in per_class_df.iterrows():
        print(f"{row['class']:<6}  {row['precision']:>9.4f}  "
              f"{row['recall']:>6.4f}  {row['f1']:>6.4f}  {row['support']:>7d}")
    print("-" * 46)
    print(f"{'Macro':<6}  {macro_p:>9.4f}  {macro_r:>6.4f}  {macro_f1:>6.4f}")

    return {
        "per_class": per_class_df,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def plot_confusion_matrix(all_labels, all_preds, class_names,
                          figsize=(14, 12), save_path=None):
    """
    Plot a row-normalised confusion matrix.

    Parameters:
    all_labels, all_preds : np.ndarray  — from compute_metrics()
    class_names           : list[str]
    figsize               : tuple
    save_path             : str or None — if given, saves the figure
    """
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        cm_norm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion matrix (row-normalised)", fontsize=13)

    # Annotate cells with the raw count
    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def _make_perturbed_loader(
        df, sigma_s, sigma_c, sigma_w, batch_size=32):
    """Build a DataLoader with fixed perturbation magnitudes."""
    class PerturbedDataset(AMLDataset):
        """AMLDataset subclass that always applies a fixed perturbation."""

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["abs_path"]).convert("RGB")
            img_np = np.array(img)
            img_np = perturb_staining(
                img_np,
                sigma_s=self.sigma_s,
                sigma_c=self.sigma_c,
                sigma_w=self.sigma_w,
                rng=self.rng,
            )
            img = self.eval_transform(Image.fromarray(img_np))
            label = self.class_to_idx[row["label"]]
            return img, label

    class_to_idx = {c: i for i, c in enumerate(
        sorted(df["label"].unique().tolist()))}

    ds = PerturbedDataset(
        df, class_to_idx,
        augment=False,
        sigma_s=sigma_s,
        sigma_c=sigma_c,
        sigma_w=sigma_w,
    )
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False, num_workers=0)


def robustness_curve(
    models_dict,
    test_df,
    class_names,
    device,
    param_name="sigma_s",
    param_values=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    fixed_sigma_s=0.0,
    fixed_sigma_c=0.0,
    fixed_sigma_w=0.0,
    batch_size=32,
    save_path=None,
):
    """
    Plot macro-F1 vs perturbation severity for one or more models.

    Parameters:
    models_dict   : dict[str, nn.Module]
    test_df       : pd.DataFrame          — test split from preprocess.py
    class_names   : list[str]
    device        : torch.device
    param_name    : str  — one of "sigma_s", "sigma_c", "sigma_w"
    param_values  : iterable of floats    — x-axis values
    fixed_*       : float                 — values held constant for the other two params
    batch_size    : int
    save_path     : str or None

    Returns:
    results : dict[str, list[float]]  — macro-F1 per model per sigma value
    """
    assert param_name in ("sigma_s", "sigma_c", "sigma_w"), \
        "param_name must be one of sigma_s, sigma_c, sigma_w"

    results = {name: [] for name in models_dict}

    for sigma in param_values:
        kwargs = dict(
            sigma_s=fixed_sigma_s,
            sigma_c=fixed_sigma_c,
            sigma_w=fixed_sigma_w,
        )
        kwargs[param_name] = sigma

        loader = _make_perturbed_loader(
            test_df, **kwargs, batch_size=batch_size)

        for model_name, model in models_dict.items():
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in loader:
                    imgs, labels = _to_device(batch, device)
                    preds = model(imgs).argmax(dim=1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            _, _, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="macro", zero_division=0
            )
            results[model_name].append(f1)
            print(
                f"  {model_name:20s}  {param_name}={sigma:.2f}  macro-F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (model_name, f1s) in enumerate(results.items()):
        ax.plot(param_values, f1s, marker="o", label=model_name,
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Macro-averaged F1", fontsize=12)
    ax.set_title(
        f"Robustness curve — macro-F1 vs {param_name}",
        fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Robustness curve saved to {save_path}")
    plt.show()

    return results


class GradCAM:
    """
    Grad-CAM for ConvNeXt-Tiny inside AMLPipeline.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(img_tensor, class_idx)
        cam.remove()
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # Hook onto the last block of ConvNeXt-Tiny's feature extractor
        target_layer = model.backbone.features[-1]

        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        """
        Compute a Grad-CAM heatmap.

        Parameters:
        x         : torch.Tensor (1, 3, H, W), values in [0, 1]
        class_idx : int or None — if None, uses the predicted class

        Returns:
        heatmap : np.ndarray (H, W), values in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients over spatial dims
        weights = self.gradients.mean(
            dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (
            weights *
            self.activations).sum(
            dim=1,
            keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, class_idx

    def remove(self):
        """Remove hooks to free memory."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def plot_gradcam(model, img_tensor, class_names, device,
                 class_idx=None, alpha=0.5, save_path=None):
    """
    Plot a single image with its Grad-CAM heatmap overlaid.

    Parameters:
    model       : AMLPipeline
    img_tensor  : torch.Tensor (1, 3, H, W), values in [0, 1]
    class_names : list[str]
    device      : torch.device
    class_idx   : int or None  — target class (None = predicted class)
    alpha       : float        — heatmap overlay transparency
    save_path   : str or None
    """
    img_tensor = img_tensor.to(device)
    cam_gen = GradCAM(model)

    heatmap, predicted_idx = cam_gen(img_tensor, class_idx)
    cam_gen.remove()

    img_np = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title(
        f"Overlay\nPredicted: {
            class_names[predicted_idx]}",
        fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM saved to {save_path}")
    plt.show()


def plot_lab_distributions(model, loader, device,
                           n_batches=10, save_path=None):
    """
    Plot a*/b* chromatic scatter distributions before and after the
    Lab-domain correction module, coloured by class.

    Parameters:
    model     : AMLPipeline
    loader    : DataLoader
    device    : torch.device
    n_batches : int — number of batches to sample
    save_path : str or None
    """
    model.eval()

    a_before_all, b_before_all = [], []
    a_after_all, b_after_all = [], []
    label_all = []

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device)

            lab_before, lab_after = model.get_corrected_lab(imgs)

            a_before_all.append(
                lab_before[:, 1].flatten().cpu().numpy())
            b_before_all.append(
                lab_before[:, 2].flatten().cpu().numpy())
            a_after_all.append(lab_after[:, 1].flatten().cpu().numpy())
            b_after_all.append(lab_after[:, 2].flatten().cpu().numpy())

            B, _, H, W = imgs.shape
            label_all.append(
                labels.unsqueeze(1).expand(
                    B, H * W).flatten().cpu().numpy()
            )

    a_before = np.concatenate(a_before_all)
    b_before = np.concatenate(b_before_all)
    a_after = np.concatenate(a_after_all)
    b_after = np.concatenate(b_after_all)
    labels = np.concatenate(label_all)

    rng = np.random.default_rng(42)
    idx = rng.choice(
        len(a_before),
        size=min(
            50_000,
            len(a_before)),
        replace=False)

    a_before, b_before = a_before[idx], b_before[idx]
    a_after, b_after = a_after[idx], b_after[idx]
    labels = labels[idx]

    cmap = plt.cm.get_cmap("tab15", 15) if hasattr(plt.cm, "tab15") \
        else plt.cm.get_cmap("tab20", 15)
    num_class = int(labels.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for c in range(num_class):
        mask = labels == c
        if mask.sum() == 0:
            continue
        color = cmap(c)
        axes[0].scatter(a_before[mask], b_before[mask],
                        s=0.5, alpha=0.3, color=color, rasterized=True)
        axes[1].scatter(a_after[mask], b_after[mask],
                        s=0.5, alpha=0.3, color=color, rasterized=True)

    for ax, title in zip(
            axes, ["Before correction", "After correction"]):
        ax.set_xlabel("a*  (green ← → magenta)", fontsize=11)
        ax.set_ylabel("b*  (blue ← → yellow)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Compute and display spread reduction
    spread_before = np.std(a_before) + np.std(b_before)
    spread_after = np.std(a_after) + np.std(b_after)
    reduction_pct = (spread_before - spread_after) / \
        (spread_before + 1e-8) * 100
    fig.suptitle(
        f"a*/b* chromatic distribution — "
        f"spread reduction: {reduction_pct:+.1f}%",
        fontsize=13,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Lab distribution plot saved to {save_path}")
    plt.show()

    print(
        f"\na* std before: {np.std(a_before):.3f}  |  after: {np.std(a_after):.3f}")
    print(
        f"b* std before: {np.std(b_before):.3f}  |  after: {np.std(b_after):.3f}")
    print(f"Combined spread reduction: {reduction_pct:+.1f}%")


def plot_training_curves(train_losses, val_losses,
                         train_accs, val_accs,
                         save_path=None):
    """
    Plot loss and accuracy curves over training epochs.

    Parameters:
    train_losses, val_losses : list[float]
    train_accs,  val_accs   : list[float]
    save_path : str or None
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train", linewidth=2)
    axes[0].plot(
        epochs,
        val_losses,
        label="Val",
        linewidth=2,
        linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train", linewidth=2)
    axes[1].plot(
        epochs,
        val_accs,
        label="Val",
        linewidth=2,
        linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.show()
