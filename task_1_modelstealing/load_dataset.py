import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("task_1_modelstealing/data/ExampleModelStealingPub.pt")

    print(dataset.ids, dataset.imgs, dataset.labels)

    
