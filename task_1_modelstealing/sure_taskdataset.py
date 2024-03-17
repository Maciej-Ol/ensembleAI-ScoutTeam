from torch.utils.data import Dataset
from typing import Tuple
import torch


class SureTaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.embeddings = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            transformed_img = self.transform(img.convert("RGB"))
        label = self.labels[index]
        return id_, img, transformed_img, label

    def __len__(self):
        return len(self.ids)