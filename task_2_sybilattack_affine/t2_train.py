import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from taskdataset import TaskDataset
import matplotlib.pyplot as plt
import os
import time
import pickle as pkl
from tqdm import tqdm
import argparse

from t2_functions import train, validate, l1_reg, lin_augment_affine

class RepresentationsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return torch.tensor(x), torch.tensor(y)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

def main(config):

    criterion = nn.MSELoss()
    batch_size = 32
    epochs = 10
    lr = 0.001

    if config.dataset == 'test':
        folder = os.path.join('task_2_sybilattack_affine/data', config.dataset, config.task, str(config.num))
        with open(f'{folder}/A_train', 'rb') as f:
            A_train_reps = pkl.load(f)
        with open(f'{folder}/B_train', 'rb') as f:
            B_train_reps = pkl.load(f)
        with open(f'{folder}/A_test', 'rb') as f:
            A_test_reps = pkl.load(f)
        with open(f'{folder}/B_test', 'rb') as f:
            B_test_reps = pkl.load(f)

        A_train_aug = torch.tensor(A_train_reps, dtype=torch.float32)
        B_train_aug = torch.tensor(B_train_reps, dtype=torch.float32)
        A_train_aug, B_train_aug = lin_augment_affine(A_train_aug, B_train_aug)

        A_test_reps = torch.tensor(A_test_reps, dtype=torch.float32)
        B_test_reps = torch.tensor(B_test_reps, dtype=torch.float32)

        train_dataset = RepresentationsDataset(x=A_train_aug, y=B_train_aug)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
        test_dataset = RepresentationsDataset(x=A_test_reps, y=B_test_reps)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        lin_net = Linear(384, 384)
        lin_empty_net = Linear(384, 384)

        optim = torch.optim.Adam(lr=lr, params=lin_net.parameters())

        lin_last_net, lin_best_net = train(epochs, optim, criterion, l1_reg, train_loader, test_loader, lin_net, lin_empty_net, reg_lambda=0.0001)

        train_loss_lin = validate(criterion, train_loader, lin_best_net)
        print(f"lin train: {train_loss_lin}")

        test_loss_lin = validate(criterion, test_loader, lin_best_net)
        print(f"lin test: {test_loss_lin}")

        B_reps = B_train_reps
        B_reps.extend(B_test_reps)
        B_reps = torch.tensor(B_reps)

        A_preds = lin_last_net(B_reps)

        print(criterion(A_preds, B_reps))

    if config.dataset == 'submit':

        preds = []
        ids = []

        for i in range(10):
            print(f"big loop: {i}")
            folder = os.path.join('task_2_sybilattack_affine/data', config.dataset, config.task, str(config.num), f'partition_{i}')
            with open(f'{folder}/A_train', 'rb') as f:
                A_train_reps = pkl.load(f)
            with open(f'{folder}/B_train', 'rb') as f:
                B_train_reps = pkl.load(f)
            with open(f'{folder}/B_test', 'rb') as f:
                B_test_reps = pkl.load(f)
            with open(f'{folder}/ids_train', 'rb') as f:
                ids_train = pkl.load(f)
            with open(f'{folder}/ids_test', 'rb') as f:
                ids_test = pkl.load(f)

            A_train_aug = torch.tensor(A_train_reps, dtype=torch.float32)
            B_train_aug = torch.tensor(B_train_reps, dtype=torch.float32)
            A_train_aug, B_train_aug = lin_augment_affine(A_train_aug, B_train_aug)

            train_dataset = RepresentationsDataset(x=A_train_aug, y=B_train_aug)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            lin_net = Linear(384, 384)
            lin_empty_net = Linear(384, 384)

            optim = torch.optim.Adam(lr=lr, params=lin_net.parameters())

            lin_last_net, lin_best_net = train(epochs, optim, criterion, l1_reg, train_loader, train_loader, lin_net, lin_empty_net, reg_lambda=0.0001)

            B_reps = np.concatenate([B_train_reps, B_test_reps], axis=0)
            B_reps = torch.tensor(B_reps, dtype=torch.float32)

            A_preds = lin_last_net(B_reps)
            A_preds = A_preds.detach().numpy()
            A_preds[:200] = np.array(A_train_reps)
            
            preds.append(A_preds)

            # to delete
            # ids_train = np.arange(B_train_reps.shape[0])
            # ids_test = np.arange(B_test_reps.shape[0])

            ids_iter = np.concatenate([ids_train, ids_test], axis=0)
            ids.append(ids_iter)

            # Convert the concatenated tensor to a NumPy array

        preds = np.concatenate(preds, axis=0)
        ids = np.concatenate(ids, axis=0)

        id_rep_map = {id: rep for id, rep in zip(ids, preds)}
        
        dataset = torch.load("task_2_sybilattack_affine/data/SybilAttack.pt")
        representations = [id_rep_map[id] for id in dataset.ids]

        np.savez("task_2_sybilattack_affine/task2_submission.npz", ids=dataset.ids, representations=representations)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num', type=int, default=1, help='unique number not to overwrite datasets')
    parser.add_argument('--task', type=str, default='affine', help='binary or affine')
    parser.add_argument('--dataset', type=str, default='submit', help='test or submit')
    
    config = parser.parse_args()

    print(config)
    main(config)