from torch.utils.data import Dataset
from typing import Tuple
import torch


class SolveDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []
        self.enocodings = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        enocoding = self.enocodings[index]
        return id_, img, label, enocoding

    def __len__(self):
        return len(self.ids)
    

class SaveSolveDataset():
    def __init__(self, path_to_vectors):

        self.ids = []
        self.imgs = []
        self.labels = []
        self.enocodings = []

        self.path_to_vectors = path_to_vectors

    def add(self, id, img, label, encoding):
        self.ids.append(id)
        self.imgs.append(img)
        self.labels.append(label)
        self.enocodings.append(encoding)

    def save(self):
        solve_dataset = SolveDataset()
        
        solve_dataset.ids = self.ids
        solve_dataset.imgs = self.imgs
        solve_dataset.labels = self.labels
        solve_dataset.enocodings = self.enocodings

        torch.save(solve_dataset, self.path_to_vectors)

