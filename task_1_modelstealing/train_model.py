import torchvision.models as models
import torch
import torch.nn as nn
from ModelToSteal import ModelToSteal, ModelToStealMockup, ModelToStealOfficial, ModelToStealRandomMockup
from torchvision import transforms
from taskdataset import TaskDataset
from sure_taskdataset import SureTaskDataset
import os

from PIL import Image

import torch.nn.functional as F

from StealingModel import StealingModel

def main():
    torch.device('cuda:0')
    args = {"epochs": 5, "log_interval": 50, "save_interval": 100}

    # args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = "cpu"
    stealing_model = build_stealing_model(args).to(args['device'])
    dataset = build_datasets(args)
    optimizer = build_optimizer(args, stealing_model)
    loss_func = build_loss_func(args)

    train(args, dataset, stealing_model, optimizer, loss_func)

def build_stealing_model(args):
    stealing_model = StealingModel()
    return stealing_model

def build_datasets(args):
    dataset = torch.load(f"./task_1_modelstealing/data/trained_0_299.pt")
    # dataset.transform = transforms.Compose([transforms.ToTensor()])

    return dataset

def build_optimizer(args, stealing_model):
    lr = 0.1
    momentum = 0.9
    weight_decay = 0.0

    return torch.optim.SGD(
        stealing_model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

def build_loss_func(args):
    return nn.MSELoss()
    # return nn.MSELoss().to(args['device'])

def train(args: int, dataset: TaskDataset, stealing_model: StealingModel, optimizer, loss_func):
    stealing_model.train()

    print("start training")
    zipped_dataset = list(zip(dataset['ids'], dataset['images'], dataset['embeddings']))
    mean=[0.3013, 0.2609, 0.2916]
    std=[0.3070, 0.2584, 0.2920]
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    for epoch in range(args["epochs"]):
        loss_sum = 0
        print(f"start epoch {epoch}")
        for batch_idx, (id, image, embedding) in enumerate(zipped_dataset):
            transformed_img = transformation(image)
            optimizer.zero_grad()
            output = stealing_model(transformed_img.reshape(1, 3, 32, 32).to(args['device'])).reshape(512)
            output = output.to(args['device'])
            # loss = loss_func(output, embbeding).to(args['device'])
            loss = loss_func(output, embedding)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if batch_idx % args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx , len(dataset['ids']),
                    100. * batch_idx / len(dataset['ids']), loss_sum))
                
        torch.onnx.export(
            stealing_model,
            torch.randn(1, 3, 32, 32),
                f"task_1_modelstealing/models/submission_{batch_idx}.onnx",
                export_params=True,
                input_names=["x"],
            )

if __name__ == "__main__":
    main()
