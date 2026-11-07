"""Local-weight ResNet U-Net for the supplied retinal vessel segmentation task.

This module never downloads weights. ``weights_path`` may point to an official
torchvision ResNet ImageNet-1K V1 state dictionary that the caller has obtained
separately. The complete classifier state is loaded strictly before the unused
classification head is discarded. Passing None is suitable when a complete
trained segmentation checkpoint will subsequently be loaded.

Input: floating RGB NCHW values in [0, 1]. ImageNet normalization happens inside
the model. Output: one vessel logit per input pixel, including odd image sizes.
"""

from hashlib import sha256
from itertools import chain
from math import gcd
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = gcd(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        return self.layers(image)


class PretrainedUNet(nn.Module):
    """ResNet18/34 encoder plus a compact decoder and full-resolution branch.

    ``freeze_bn`` freezes encoder running statistics, not encoder gradients or
    BatchNorm affine parameters. The optimizer can therefore fine-tune the
    complete encoder while avoiding unstable batch statistics on small crops.
    """

    ARCHITECTURE = "resnet_unet_fullres_gn_v1"

    def __init__(self, encoder_name="resnet18", weights_path=None, freeze_bn=True):
        super().__init__()
        builders = {"resnet18": resnet18, "resnet34": resnet34}
        if encoder_name not in builders:
            raise ValueError("encoder_name must be 'resnet18' or 'resnet34'")
        self.encoder_name = str(encoder_name)
        self.freeze_bn = bool(freeze_bn)
        self.weights_path = None
        self.weights_sha256 = None

        # Do not pass a weights enum here: that would enable a network fetch.
        backbone = builders[self.encoder_name](weights=None)
        if weights_path is not None:
            path = Path(weights_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Local encoder weights not found: {path}")
            state = torch.load(path, map_location="cpu", weights_only=True)
            # Load fc as well, so missing classifier keys or a mismatched
            # backbone cannot silently pass as an official encoder checkpoint.
            backbone.load_state_dict(state, strict=True)
            self.weights_path = str(path)
            digest = sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.weights_sha256 = digest.hexdigest()

        self.encoder = nn.ModuleDict({
            "conv1": backbone.conv1,
            "bn1": backbone.bn1,
            "relu": backbone.relu,
            "maxpool": backbone.maxpool,
            "layer1": backbone.layer1,
            "layer2": backbone.layer2,
            "layer3": backbone.layer3,
            "layer4": backbone.layer4,
        })
        # The decoder keeps a native-resolution RGB route for fine vessels.
        self.full_resolution = _DoubleConv(3, 16)
        self.decoders = nn.ModuleList([
            _DoubleConv(512 + 256, 128),  # layer4 -> layer3, 1/16 scale
            _DoubleConv(128 + 128, 96),   # -> layer2, 1/8 scale
            _DoubleConv(96 + 64, 64),     # -> layer1, 1/4 scale
            _DoubleConv(64 + 64, 32),     # -> stem, 1/2 scale
            _DoubleConv(32 + 16, 16),     # -> native RGB feature map
        ])
        self.output = nn.Conv2d(16, 1, 1)
        # Initialize only new decoder modules; never overwrite loaded encoder
        # weights with a model-wide .apply(initializer) call.
        for module in (self.full_resolution, self.decoders):
            module.apply(self._initialize_decoder)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        nn.init.zeros_(self.output.bias)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.train(self.training)

    @staticmethod
    def _initialize_decoder(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def train(self, mode=True):
        """Keep pretrained BatchNorm statistics fixed through mode changes."""
        super().train(mode)
        if self.freeze_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def encoder_parameters(self):
        """Parameters for the encoder learning-rate group (all trainable)."""
        return self.encoder.parameters()

    def decoder_parameters(self):
        """All non-encoder parameters, with no overlap or omitted head weights."""
        return chain(self.full_resolution.parameters(), self.decoders.parameters(), self.output.parameters())

    def describe(self):
        """Architecture and initial-weight provenance, independent of metrics.

        ``initialization`` describes construction only. A later load of a full
        segmentation checkpoint does not change these provenance fields; the
        training/checkpoint manifest should retain its original description.
        """
        encoder_count = sum(parameter.numel() for parameter in self.encoder_parameters())
        decoder_count = sum(parameter.numel() for parameter in self.decoder_parameters())
        return {
            "architecture": self.ARCHITECTURE,
            "encoder_name": self.encoder_name,
            "initialization": "local_encoder_state_dict" if self.weights_path else "random_encoder",
            "pretrained_encoder_loaded": self.weights_path is not None,
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "automatic_downloads": False,
            "freeze_encoder_bn_statistics": self.freeze_bn,
            "encoder_parameters_trainable": all(p.requires_grad for p in self.encoder_parameters()),
            "encoder_parameters": encoder_count,
            "decoder_parameters": decoder_count,
            "total_parameters": encoder_count + decoder_count,
            "decoder_channels": [128, 96, 64, 32, 16],
            "full_resolution_channels": 16,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "input": "RGB float NCHW in [0,1]",
            "output": "one-channel vessel logits at input spatial size",
        }

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 1:
            raise ValueError("Expected non-empty N x 3 x H x W RGB input")
        if not image.is_floating_point():
            raise TypeError("Input must be floating RGB values in [0,1]")
        normalized = (image - self.input_mean) / self.input_std
        full = self.full_resolution(normalized)
        encoder = self.encoder
        stem = encoder["relu"](encoder["bn1"](encoder["conv1"](normalized)))
        first = encoder["layer1"](encoder["maxpool"](stem))
        second = encoder["layer2"](first)
        third = encoder["layer3"](second)
        value = encoder["layer4"](third)
        for decoder, skip in zip(self.decoders, (third, second, first, stem, full)):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat((value, skip), dim=1))
        return self.output(value)
