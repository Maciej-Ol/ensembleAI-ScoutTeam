from torch.utils.data import Dataset
from typing import Tuple
import torch
import random


class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

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

    def shuffle(self):
        random.seed(1241251)

        # Generate a shuffling order
        shuffling_order = list(range(len(self.ids)))
        random.shuffle(shuffling_order)

        # Shuffle all three lists using the same order
        self.ids = [self.ids[i] for i in shuffling_order]
        self.imgs = [self.imgs[i] for i in shuffling_order]
        self.labels = [self.labels[i] for i in shuffling_order]
