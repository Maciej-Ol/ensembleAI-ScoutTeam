import numpy as np
import torch
from endpoints.requests import sybil_submit
def load_data():
    load_dataset = np.load("task_2_sybilattack/data/full_train_data2.npz")
    return load_dataset

def forward_all():
    load_dataset = load_data()
    result = []
    for i in range(0,10):
        PATH = f"task_2_sybilattack/models2/submission{i}"
        model = torch.load(PATH)
        data = load_dataset['repr_def'][i*2000:(i+1)*2000]

        output = model(data)

        if len(result) ==0:
            result = output
        else:
            result = np.concatenate((result,output))

    
    np.savez(f"task_2_sybilattack/result.npz",
        ids = load_data.ids,
        representation = result)
    sybil_submit("binary",f"task_2_sybilattack/result.npz")