"""Dataset and image transforms."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ImageNet statistics: used for BOTH models so the input distribution is identical.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, size: int = 224):
    if train:
        # light augmentation, training only. No horizontal flip: the heart's side
        # carries meaning in chest X-rays.
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class XrayDataset(Dataset):
    def __init__(self, csv_path, root, transform):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # X-rays are grayscale; convert("RGB") copies the channel 3 times so the
        # image fits ImageNet-style 3-channel models.
        img = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])

    @property
    def labels(self):
        return self.df["label"].to_numpy()
