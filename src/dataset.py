import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class FishDataset(Dataset):
    def __init__(self, csv_path, seg_root, size=(256, 448)):
        self.rows = pd.read_csv(csv_path)
        self.seg_root = Path(seg_root)
        self.size = size

    def __len__(self):
        return len(self.rows)

    def _paths(self, row):
        rel = row["ID"]
        image = self.seg_root / "images" / f"{rel}.jpg"
        mask = self.seg_root / "masks" / f"{rel}.png"
        return image, mask

    def __getitem__(self, idx):
        """Resize the image and mask to the specified size and return them as tensors."""
        # Get the row corresponding to the index
        row = self.rows.iloc[idx]
        image_path, mask_path = self._paths(row)

        # Open the image and mask in corresponding modes
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize the image and mask to the specified size
        image = image.resize((self.size[1], self.size[0]))
        mask = mask.resize((self.size[1], self.size[0]), Image.NEAREST)

        # Convert the image and mask to tensors
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask))
        mask = (mask > 127).long()  # Convert mask to binary (0 or 1)

        return image, mask
