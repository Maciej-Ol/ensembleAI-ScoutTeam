import torchvision.models as models
import torch
# from endpoints.model_stealing import model_stealing
from  taskdataset import TaskDataset
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import MLP

from PIL import Image

class StealingModelTask2(nn.Module):

    def __init__(self, out_dim=512):
        super(StealingModelTask2, self).__init__()

        self.layers = nn.Sequential(
            MLP(384, [1024, 1024, 384])
        )
        # self.resnet = models.resnet18(weights=None)
        # self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)



    def forward(self, x):
        x = self.layers(x)

        return x
