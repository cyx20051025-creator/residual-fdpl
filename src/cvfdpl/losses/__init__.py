"""Training objectives."""

from cvfdpl.losses.fdpl import (
    BrightnessAwareLoss,
    OriginalFDPLoss,
    PerChannelLogFDPLoss,
    VGGPerceptualLoss,
)

__all__ = [
    "BrightnessAwareLoss",
    "OriginalFDPLoss",
    "PerChannelLogFDPLoss",
    "VGGPerceptualLoss",
]

