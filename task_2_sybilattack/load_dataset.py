import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("task_2_sybilattack/data/ExampleSybilAttack.pt")

    print(dataset.ids, dataset.imgs, dataset.labels)
