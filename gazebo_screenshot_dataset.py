from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import random


class GazeboScreenshotDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = sorted(
            [f for f in os.listdir(self.root_dir) if f.endswith(".png")]
        )
        random.Random(1337).shuffle(self.filenames)
        split_idx = 3 * len(self.filenames) // 4
        if mode == "valid":
            self.filenames = self.filenames[split_idx:]
        elif mode == "train":
            self.filenames = self.filenames[:split_idx]
        else:
            print(
                "Warning: using entire dataset, because mode was neither 'train' nor 'valid'"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename_rgb = os.path.join(self.root_dir, self.filenames[idx])
        filename_label = filename_rgb.replace("/images/", "/labels/")

        rgb = cv2.imread(filename_rgb).astype(np.float32)[:, :, ::-1] / 255.0
        label_image = cv2.imread(filename_label)

        weeds = np.linalg.norm(label_image - [0, 0, 255], axis=-1) < 40
        maize = np.linalg.norm(label_image - [0, 255, 0], axis=-1) < 40
        trash = np.linalg.norm(label_image - [255, 0, 0], axis=-1) < 40
        background = (weeds + maize + trash) == 0

        onehot = np.stack([background, maize, weeds, trash], axis=-1)

        labels = onehot.argmax(axis=-1)

        sample = {"x": rgb, "y": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x, y = sample["x"], sample["y"]
        x = x.swapaxes(0, 2)
        y = y.swapaxes(0, 1)
        # return {'x': torch.from_numpy(x),
        #        'y': torch.from_numpy(y)}
        return (
            torch.tensor(x, dtype=torch.float, device=device),
            torch.tensor(y, dtype=torch.long, device=device),
        )
