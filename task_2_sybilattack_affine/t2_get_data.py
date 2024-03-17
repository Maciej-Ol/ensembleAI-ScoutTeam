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
import argparse

from t2_functions import partition_ids, find_independent_set

import sys

sys.path.append(os.path.join(os.getcwd(), "task_2_sybilattack/"))

from task_2_sybilattack_affine.endpoints.requests2 import sybil, sybil_reset

def main(config):
    print(os.getcwd())
    dataset = torch.load("task_2_sybilattack/data/SybilAttack.pt")

    ids = np.array(dataset.ids)
    binned_ids = partition_ids(ids, main_bin_num=10)

    task = config.task

    sybil_reset(home_or_defense='home', binary_or_affine=task)
    sybil_reset(home_or_defense='defense', binary_or_affine=task)
    time.sleep(10)

    if config.dataset == 'test':
        ids = binned_ids[0]
        print(f"querying endpoint B")
        B_reps = sybil(ids=ids,
                       home_or_defense='defense',
                       binary_or_affine=task)

        B_indep, indexes, success = find_independent_set(B_reps, k=200)
        print(f"success: {success}")
        print(indexes)

        ids_train = ids[indexes]
        mask = np.ones(len(ids), dtype=bool)
        mask[indexes] = False
        ids_test = ids[mask]

        B_reps = np.array(B_reps)
        B_train_reps = B_reps[indexes]
        B_test_reps = B_reps[mask]
        print(f"guerying endpoint A for {len(ids_train) + len(ids_test)} total")
        A_train_reps = sybil(ids=ids_train,
                        home_or_defense='home',
                        binary_or_affine=task)
        print(f"A train reps: {len(A_train_reps)}")

        A_test_reps = sybil(ids=ids_test,
                        home_or_defense='home',
                        binary_or_affine=task)
        print(f"A test reps: {len(A_test_reps)}")

        folder = os.path.join('task_2_sybilattack/data', config.dataset, task, str(config.num))
        os.makedirs(folder, exist_ok=True)

        with open(f'{folder}/A_train', 'wb') as f:
            pkl.dump(A_train_reps, f)
        with open(f'{folder}/B_train', 'wb') as f:
            pkl.dump(B_train_reps, f)
        with open(f'{folder}/A_test', 'wb') as f:
            pkl.dump(A_test_reps, f)
        with open(f'{folder}/B_test', 'wb') as f:
            pkl.dump(B_test_reps, f)

    if config.dataset == 'submit':
        for i in range(1, 10):
            print(f'main loop: {i}')
            sybil_reset(home_or_defense='defense', binary_or_affine=task)
            time.sleep(10)
            ids = binned_ids[i]
            print(len(ids))
            B_reps = sybil(ids=ids,
                        home_or_defense='defense',
                        binary_or_affine=task)
            
            B_indep, indexes, success = find_independent_set(B_reps, k=200)
            print(f"success: {success}")

            ids_train = ids[indexes]
            mask = np.ones(len(ids), dtype=bool)
            mask[indexes] = False
            ids_test = ids[mask]

            B_reps = np.array(B_reps)
            B_train_reps = B_reps[indexes]
            B_test_reps = B_reps[mask]

            A_train_reps = sybil(ids=ids_train,
                            home_or_defense='home',
                            binary_or_affine=task)
            print(f"A train reps: {len(A_train_reps)}")

            folder = os.path.join('task_2_sybilattack/data', config.dataset, task, str(config.num), f'partition_{i}')
            os.makedirs(folder, exist_ok=True)

            with open(f'{folder}/A_train', 'wb') as f:
                pkl.dump(A_train_reps, f)
            with open(f'{folder}/B_train', 'wb') as f:
                pkl.dump(B_train_reps, f)
            with open(f'{folder}/B_test', 'wb') as f:
                pkl.dump(B_test_reps, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num', type=int, default=0, help='unique number not to overwrite datasets')
    parser.add_argument('--task', type=str, default='affine', help='binary or affine')
    parser.add_argument('--dataset', type=str, default='test', help='test or submit')
    
    config = parser.parse_args()

    print(config)
    main(config)