import torch
from ModelToSteal import ModelToStealOfficial, ModelToStealMockup
from data_steal.solvedataset import SaveSolveDataset
from  taskdataset import TaskDataset

data_path = "./data/ModelStealingPub.pt"
N = 100

class DataStealer():
    def __init__(self, model_to_steal, path_to_images, path_to_vectors):
        self.model_to_steal = model_to_steal
        self.path_to_images = path_to_images
        
        self.iterations_to_denoise = 1
        self.save_solve_dataset = SaveSolveDataset(path_to_vectors)

    def prepare_dataloader(self):
        # TODO Popracuj nad data augmentation
        dataset = torch.load(self.path_to_images)

        # return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset


    def get_one_embedding(self, id, img, label):
        # TODO Zrób denoise
        encoding = self.model_to_steal.get_embeddings(img)
        self.save_solve_dataset.add(id, img, label, encoding)

    def encode_data(self):
        dataloader = self.prepare_dataloader()

        id, img,label = dataloader[0]
        self.get_one_embedding(id, img, label)
        # for i, (id, img, label) in enumerate(dataloader):
        #     self.get_one_embedding(id, img, label)

        self.save_solve_dataset.save()