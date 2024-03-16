import torchvision.models as models
import torch
from endpoints.model_stealing import model_stealing
from  taskdataset import TaskDataset
import torch.nn as nn
from torchvision import transforms

from PIL import Image

class StealingModel(nn.Module):

    def __init__(self, base_model="resnet50", out_dim=512, loss=None, include_mlp=False):
        super(StealingModel, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34":  models.resnet34(pretrained=False, num_classes=out_dim)
                            ,"resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.last_layer = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.include_mlp = include_mlp
        self.loss = loss
        dim_mlp = self.backbone.fc.in_features
        self.transform = transforms.Compose([transforms.Resize(224)])

        if self.loss == "symmetrized":
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.BatchNorm1d(dim_mlp),
                                             nn.ReLU(inplace=True),
                                             self.backbone.fc)
        else:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                             nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.transform(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_layer(x)
        if self.include_mlp:
            x = self.backbone.fc(x)
        return x