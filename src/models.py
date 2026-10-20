"""ResNet-18 builders for each experimental condition."""
import torch.nn as nn
from torchvision import models


def build_model(init: str, mode: str, num_classes: int = 2):
    """
    init: 'scratch'    -> random weights
          'pretrained' -> ImageNet weights
    mode: 'full'    -> train every layer
          'partial' -> train only layer4 + classifier head (rest frozen)
          'frozen'  -> train only the classifier head (backbone = feature extractor)
    """
    assert init in ("scratch", "pretrained")
    assert mode in ("full", "partial", "frozen")
    if init == "scratch" and mode != "full":
        raise ValueError("Freezing random weights makes no sense; use mode='full' for scratch.")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if init == "pretrained" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head, always trainable

    if mode in ("partial", "frozen"):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if mode == "partial":
            for p in model.layer4.parameters():
                p.requires_grad = True

    return model


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
