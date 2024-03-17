import torchvision.models as models
import torch
import torch.nn as nn
# from ModelToSteal import ModelToSteal, ModelToStealMockup, ModelToStealOfficial, ModelToStealRandomMockup
from torchvision import transforms
from taskdataset import TaskDataset
# from sure_taskdataset import SureTaskDataset
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

from PIL import Image

import torch.nn.functional as F

from StealingModel import StealingModelTask2

# onnx_load_model_path = "submission_1.onnx"

# load_model_path = "task_1_modelstealing/models/submission_47"
TRAIN_DATA_SIZE = 200
DEF_DATA_SIZE = 2000

def main():
    torch.device('cuda:0')
    args = {"epochs": 100, "log_interval": 10, "save_interval": 100}

    # args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = "cpu"
    for training_number in range(0, 10):
        stealing_model = build_stealing_model(args).to(args['device'])
        train_dataloader, test_dataloader = build_loaders(training_number)
        optimizer = build_adam_optimizer(args, stealing_model)
        loss_func = build_loss_func(args)

        print(training_number)
        train(args, train_dataloader, test_dataloader, stealing_model, optimizer, loss_func, training_number)

def build_stealing_model(args):
    stealing_model = StealingModelTask2()

    return stealing_model


def build_loaders(training_number):
    load_dataset = np.load("task_2_sybilattack/data/full_train_data2.npz")
    threshold_home_left =  training_number * TRAIN_DATA_SIZE 
    threshold_home_right =  training_number * TRAIN_DATA_SIZE + TRAIN_DATA_SIZE

    threshold_def_left =  training_number * DEF_DATA_SIZE 
    threshold_def_right =  training_number * DEF_DATA_SIZE + TRAIN_DATA_SIZE

    len_dataset = len(load_dataset['repr_home'])
    print(f'len load_dataset[repr_home] = {len_dataset}')
    dataset = {}

    # dataset['ids'] = torch.Tensor(load_dataset['ids'][threshold_left:threshold_right])
    dataset['repr_def'] = torch.Tensor(load_dataset['repr_def'][threshold_def_left:threshold_def_right]) 
    dataset['repr_home'] = torch.Tensor(load_dataset['repr_home'][threshold_home_left:threshold_home_right]) 

    list_of_inputs = dataset["repr_def"]  # Replace with your list of PIL images

    # Apply the transformation to each image in the list and stack them into a single tensor
    # inputs_tensor = torch.stack([transform(inp) for inp in list_of_inputs], dim=0)
    inputs_tensor = torch.stack([inp for inp in list_of_inputs], dim=0)
    print(dataset["repr_home"].shape)
    outputs_tensor = torch.stack([out for out in dataset["repr_home"]], dim=0)

    dataset = TensorDataset(inputs_tensor, outputs_tensor)

    # Define the size of the train and validation sets
    train_size = int(0.9 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation

    # Split dataset into train and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Define batch size and other DataLoader parameters
    batch_size = 32
    shuffle = True  # Or False, depending on your preference

    # Create DataLoader for training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # Create DataLoader for validation set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_SGD_optimizer(args, stealing_model):
    lr = 0.1
    momentum = 0.9
    weight_decay = 0.0

    return torch.optim.SGD(
        stealing_model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

def build_adam_optimizer(args, stealing_model):
    lr = 1e-3
    return torch.optim.Adam(stealing_model.parameters(), lr=lr)

def build_loss_func(args):
    return nn.MSELoss()
    # return nn.MSELoss().to(args['device'])

def train(args: int,  train_dataloader, test_dataloader, stealing_model: StealingModelTask2, optimizer, loss_func, training_number):
    stealing_model.train()
    
    losses = []

    for epoch in range(args["epochs"]):
        loss_sum = 0
        print(f"start epoch {epoch}")
        for batch_idx, (inp, embedding) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # print(f'inp.shape: {inp.shape}')
            output = stealing_model(inp.to(args['device'])).reshape(-1, 384).to(args['device'])
            embedding.to(args['device'])
            # print(f'embedding.shape: {embedding.shape}')
            # print(f'output.shape: {output.shape}')
            loss = loss_func(output, embedding)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if batch_idx % args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx , len(train_dataloader),
                    100. * batch_idx / len(train_dataloader), loss_sum))
    

        losses.append(validate(args, loss_func, test_dataloader, stealing_model))
        print("Test Loss ", losses[-1])

    torch.save({
        'epoch': 200,
        'model_state_dict': stealing_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1],
    }, f"task_2_sybilattack/models2/submission{training_number}")
        # print(f"task_1_modelstealing/models/submission_{epoch}: Saved") 

        # torch.onnx.export(
        #     stealing_model,
        #     torch.randn(1, 384),
        #         f"task_2_sybilattack/models/submission1{training_number}_{epoch}.onnx",
        #         export_params=True,
        #         input_names=["x"],
        #     )

def test_model(args, stealing_model, test_dataloader, loss_func):
    stealing_model.val()
    loss = 0
    weights = 0
    for batch_idx, (image, embedding) in enumerate(test_dataloader):
        
        output = stealing_model(image.to(args['device'])).reshape(-1, 512).to(args['device'])
        embedding.to(args['device'])
        
        loss += loss_func(output, embedding) * embedding.shape[0]
        weights += embedding.shape[0]

    
    stealing_model.train()


def validate(args, criterion, loader, net):
    net.eval()

    with torch.no_grad():
        true = []
        pred = []
        for x, y in loader:
            x = x.to(args["device"])
            y_hat = net(x)
            true.append(y.numpy())
            pred.append(y_hat.cpu().detach().numpy())

        true = torch.tensor(np.concatenate(true, axis=0))
        pred = torch.tensor(np.concatenate(pred, axis=0))

        loss = criterion(pred, true)

    return loss

if __name__ == "__main__":
    main()
