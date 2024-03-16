import torchvision.models as models
import torch
import torch.nn as nn
from ModelToSteal import ModelToSteal, ModelToStealMockup, ModelToStealOfficial, ModelToStealRandomMockup
from torchvision import transforms
from taskdataset import TaskDataset
from sure_taskdataset import SureTaskDataset

import torch.nn.functional as F

from StealingModel import StealingModel

def main():
    torch.device('cuda:0')
    args = {"epochs": 5, "log_interval": 1, "save_interval": 100, "check_noise_interval": 50}

    # args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = "cpu"
    stealing_model = build_stealing_model(args).to(args['device'])
    victim_model = build_victim_model(args)
    dataset, sure_dataset = build_datasets(args)
    optimizer = build_optimizer(args, stealing_model)
    loss_func = build_loss_func(args)

    dataset.shuffle()

    train(args, dataset, sure_dataset, stealing_model, victim_model, optimizer, loss_func)

def build_victim_model(args) -> ModelToSteal:
    return ModelToStealOfficial()
    # return ModelToStealRandomMockup()

def build_stealing_model(args):
    stealing_model = StealingModel()

    return stealing_model

def build_datasets(args):
    dataset: TaskDataset = torch.load("task_1_modelstealing/data/ModelStealingPub.pt")
    dataset.transform = transforms.Compose([transforms.ToTensor()])

    sure_dataset: SureTaskDataset = torch.load("task_1_modelstealing/data/sure/sure.pt")

    return dataset, sure_dataset

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

def train(args: int, dataset: TaskDataset, sure_dataset, stealing_model: StealingModel, victim_model: ModelToSteal, optimizer, loss_func):
    stealing_model.train()
    print("start training")
    for epoch in range(args["epochs"]):
        print(f"start epoch {epoch}")
        for batch_idx, (id, image, transformed_img, label) in enumerate(dataset):
            if batch_idx > 200:
                break
            # image = image.to(args.device)
            image = image.convert("RGB")
            # embbeding = torch.tensor(victim_model.get_embeddings(image, id)).to(args['device'])
            print(f"start get denoised")
            # embbeding = torch.tensor(victim_model.get_denoised_embedding(image, id)).to(args['device'])
            embbeding = torch.tensor(victim_model.get_embeddings(image, id)).to(args['device'])
            print(f"end get denoised")

            file_path = f'task_1_modelstealing/data/emb/epoch_{epoch}_{id}.pt'

            # Save the tensor to the file
            torch.save(embbeding, file_path)

            optimizer.zero_grad()
            output = stealing_model(transformed_img.reshape(1, 3, 32, 32).to(args['device'])).reshape(512)
            output = output.to(args['device'])
            # loss = loss_func(output, embbeding).to(args['device'])
            loss = loss_func(output, embbeding)
            # loss = F.nll_loss(output, embbeding)
            loss.backward()
            optimizer.step()
            if batch_idx % args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx , len(dataset),
                    100. * batch_idx / len(dataset), loss.item()))
                
            
            if batch_idx % args["check_noise_interval"] == 0:
                image = sure_dataset["images"][0]
                embedding = sure_dataset["embeddings"][0]
                i = victim_model.estimate_noise(image, embedding)
                print(f'{i} iterations to denoise')

            if (batch_idx + 1) % args["save_interval"]  == 0:
                torch.onnx.export(
                    stealing_model,
                    torch.randn(1, 3, 32, 32),
                    f"task_1_modelstealing/models/submission_{batch_idx}_{epoch}.onnx",
                    export_params=True,
                    input_names=["x"],
                )
        torch.onnx.export(
            stealing_model,
            torch.randn(1, 3, 32, 32),
            f"task_1_modelstealing/modelstealing/models/submission{epoch}.onnx",
            export_params=True,
            input_names=["x"],
        )
if __name__ == "__main__":
    main()
