"""ResNet-18 builders for each experimental condition."""
import torch.nn as nn
from torchvision import models

# which residual stages are trainable in each mode (the 2-way classifier head is always trainable)
#   full        every layer, including the stem (conv1/bn1)
#   frozen      head only (backbone = fixed feature extractor)
#   partial     layer4 + head
#   partial34   layer3-4 + head   \
#   partial234  layer2-4 + head    } layer-wise freezing sweep
#   partial1234 layer1-4 + head   /  (everything except the stem)
TRAINABLE_BLOCKS = {
    "full": None,
    "frozen": (),
    "partial": ("layer4",),
    "partial34": ("layer3", "layer4"),
    "partial234": ("layer2", "layer3", "layer4"),
    "partial1234": ("layer1", "layer2", "layer3", "layer4"),
}
MODES = tuple(TRAINABLE_BLOCKS)


def build_model(init: str, mode: str, num_classes: int = 2):
    """
    init: 'scratch'    -> random weights
          'pretrained' -> ImageNet weights
    mode: one of MODES (see TRAINABLE_BLOCKS above)
    """
    assert init in ("scratch", "pretrained")
    assert mode in TRAINABLE_BLOCKS, f"unknown mode {mode}; choose from {MODES}"
    if init == "scratch" and mode != "full":
        raise ValueError("Freezing random weights makes no sense; use mode='full' for scratch.")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if init == "pretrained" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head, always trainable

    blocks = TRAINABLE_BLOCKS[mode]
    if blocks is not None:  # not 'full'
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        for name in blocks:
            for p in getattr(model, name).parameters():
                p.requires_grad = True

    return model


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
