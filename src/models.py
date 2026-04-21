"""
models.py

Neural network definitions for the AML stain correction + classification pipeline.

Architecture:
    Input [0, 1] RGB
    * RGBToLab            (differentiable, no params, from preprocess.py)
    * CorrectionModule    (flat residual encoder-decoder, Lab to Lab)
    * LabToRGB            (differentiable, no params, from preprocess.py)
    * ImageNet Normalize
    * ConvNeXt-Tiny       (pretrained, head replaced for 15-class output)
    * 15-class prediction

The full pipeline (AMLPipeline) is trained end-to-end. Gradients flow through
the Lab transforms into the correction module and into ConvNeXt-Tiny.

Usage:
    from models import AMLPipeline, build_model
    model = build_model(num_classes=15)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from preprocess import RGBToLab, LabToRGB, IMAGENET_MEAN, IMAGENET_STD


# 0.  Device helper

def get_device():
    """
    Return the best available device.
    Prints which device will be used so it is visible in the notebook.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"GPU available: {name} — training on CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS available — training on MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU found — training on CPU.")
    return device

# 1.  Flat residual correction module  (Lab to Lab)


class ResidualBlock(nn.Module):
    """
    Single residual block: Conv -> BN -> ReLU -> Conv -> BN, plus skip.

    Both convolutions preserve spatial resolution (stride=1, same padding).

    Parameters
    ----------
    channels : int — number of feature channels (in = out)
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class CorrectionModule(nn.Module):
    """
    Flat residual stain correction network operating in CIE L*a*b* space.

    Parameters
    ----------
    base_channels : int — width of the intermediate feature space (default 32)
    num_blocks    : int — number of residual blocks (default 3)
    """

    def __init__(self, base_channels=32, num_blocks=3):
        super().__init__()

        self.input_proj = nn.Conv2d(
            3, base_channels, kernel_size=1, bias=False)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Conv2d(
            base_channels, 3, kernel_size=1, bias=False)

        nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-3)

    def forward(self, lab):
        """
        Parameters
        ----------
        lab : torch.Tensor (B, 3, H, W) — CIE L*a*b* image

        Returns
        -------
        torch.Tensor (B, 3, H, W) — corrected L*a*b* image
        """
        features = self.input_proj(lab)
        features = self.res_blocks(features)
        correction = self.output_proj(features)
        return lab + correction      # global residual skip


# 2.  Full pipeline: Lab correction + ConvNeXt-Tiny

class AMLPipeline(nn.Module):
    """
    End-to-end pipeline for AML white blood cell classification.

    Parameters
    ----------
    num_classes   : int  — number of output classes (15 for this dataset)
    base_channels : int  — width of the correction module
    num_blocks    : int  — depth of the correction module
    dropout       : float — dropout rate before the final linear layer
    pretrained    : bool  — load ImageNet-pretrained ConvNeXt-Tiny weights
    """

    def __init__(
        self,
        num_classes=15,
        base_channels=32,
        num_blocks=3,
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()

        # Physical color space layers
        self.rgb_to_lab = RGBToLab()
        self.lab_to_rgb = LabToRGB()

        # Stain correction module
        self.correction = CorrectionModule(
            base_channels=base_channels,
            num_blocks=num_blocks,
        )

        # ImageNet normalisation constants
        self.register_buffer(
            "_img_mean",
            torch.tensor(
                IMAGENET_MEAN,
                dtype=torch.float32).view(
                1,
                3,
                1,
                1),
        )
        self.register_buffer(
            "_img_std",
            torch.tensor(
                IMAGENET_STD,
                dtype=torch.float32).view(
                1,
                3,
                1,
                1),
        )

        # ConvNeXt-Tiny backbone
        weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.convnext_tiny(weights=weights)

        # Replace the classifier head:
        #   original: AdaptiveAvgPool -> Flatten -> LayerNorm -> Linear
        # proposed:     AdaptiveAvgPool -> Flatten -> LayerNorm ->
        # Dropout -> Linear (num_classes)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor (B, 3, H, W), values in [0, 1]  — raw sRGB images

        Returns
        -------
        logits : torch.Tensor (B, num_classes)
        """
        # Note from Shihab:
        # Under autocast the CorrectionModule's Conv2d layers run in
        # float16.  The backward pass through LabToRGB produces large
        # gradients for dark pixels, and Lab-space activations
        # reach very high values.  Multiplied together in float16, these
        # exceed float16 max resulting in NaNs.  The GradScaler then
        # detects these NaNs and skips the entire optimizer step for all params,
        # so the model never learns.
        # Fix: disable autocast for the Lab section.

        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x.float()
            lab = self.rgb_to_lab(x_f32)
            lab_corrected = self.correction(lab)
            rgb_corrected = self.lab_to_rgb(lab_corrected)

        rgb_norm = (rgb_corrected - self._img_mean) / self._img_std
        logits = self.backbone(rgb_norm)
        return logits

    def get_corrected_lab(self, x):
        """
        Returns the corrected Lab tensor for a batch.
        Used in helpers.py for visualising a*/b* chromatic distributions.

        Parameters
        ----------
        x : torch.Tensor (B, 3, H, W), values in [0, 1]

        Returns
        -------
        lab_input     : torch.Tensor (B, 3, H, W) — Lab before correction
        lab_corrected : torch.Tensor (B, 3, H, W) — Lab after correction
        """
        with torch.no_grad():
            lab_input = self.rgb_to_lab(x)
            lab_corrected = self.correction(lab_input)
        return lab_input, lab_corrected

    def compute_consistency_loss(self, x_perturbed, x_clean):
        """
        Stain consistency loss: the correction module should map both the
        clean and stain-perturbed versions of the same image to the same
        corrected Lab representation.

        Parameters
        ----------
        x_perturbed : torch.Tensor (B, 3, H, W), values in [0, 1]
        x_clean     : torch.Tensor (B, 3, H, W), values in [0, 1]

        Returns
        -------
        loss : torch.Tensor (scalar) — mean L1 distance in Lab space
        """
        with torch.amp.autocast('cuda', enabled=False):
            x_p = x_perturbed.float()
            x_c = x_clean.float()

            lab_corrected_p = self.correction(self.rgb_to_lab(x_p))
            lab_corrected_c = self.correction(self.rgb_to_lab(x_c))

        return nn.functional.l1_loss(lab_corrected_p, lab_corrected_c)


# 3.  Builder function

def build_model(
    num_classes=15,
    base_channels=32,
    num_blocks=3,
    dropout=0.3,
    pretrained=True,
):
    """
    Instantiate AMLPipeline, move to the best available device, and print
    a parameter summary.

    Returns
    -------
    model  : AMLPipeline
    device : torch.device
    """
    device = get_device()
    model = AMLPipeline(
        num_classes=num_classes,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        pretrained=pretrained,
    ).to(device)

    # Parameter summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)
    correction = sum(p.numel() for p in model.correction.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())

    print(f"\nModel summary:")
    print(f"  Correction module params : {correction:>10,}")
    print(f"  ConvNeXt-Tiny params     : {backbone:>10,}")
    print(f"  Total params             : {total:>10,}")
    print(f"  Trainable params         : {trainable:>10,}")

    return model, device
