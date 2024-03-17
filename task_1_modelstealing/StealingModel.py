import torchvision.models as models
import torch
from endpoints.model_stealing import model_stealing
from  taskdataset import TaskDataset
import torch.nn as nn
from torchvision import transforms

from PIL import Image

class StealingModel(nn.Module):

    def __init__(self, out_dim=512):
        super(StealingModel, self).__init__()
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
        #                     "resnet34":  models.resnet34(pretrained=False, num_classes=out_dim)
        #                     ,"resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = models.resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.fc = nn.Linear(2048, 512)

        # self.include_mlp = include_mlp
        # self.loss = loss
        # dim_mlp = self.backbone.fc.in_features
        # self.transform = transforms.Compose([transforms.Resize(224)])

        # if self.loss == "symmetrized":
        #     self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
        #                                      nn.BatchNorm1d(dim_mlp),
        #                                      nn.ReLU(inplace=True),
        #                                      self.backbone.fc)
        # else:
        #     self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
        #                                      nn.ReLU(), self.backbone.fc)

    # def _get_basemodel(self, model_name):
    #     try:
    #         model = self.resnet_dict[model_name]
    #     except KeyError:
    #         raise Exception(
    #             "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
    #     else:
    #         return model

    def forward(self, x):
        # x = self.transform(x)

        x = self.backbone(x)

        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
        # x = self.backbone.avgpool(x)
        # x = torch.flatten(x, 1)
        # # x = self.last_layer(x)
        # # if self.include_mlp:
        # x = self.backbone.fc(x)

        return x

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(3*32*32, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet-50 model
        resnet50 = models.resnet50(pretrained=True)
        # Remove the last fully connected layer (classifier)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        # Set ResNet-50 to evaluation mode
        self.feature_extractor.eval()
        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # Add an additional linear layer to reduce the dimensionality to 512
        self.linear_layer = nn.Linear(2048, 512)  # ResNet-50's output dimension is 2048

    def forward(self, x):
        # Extract features using ResNet-50
        with torch.no_grad():
            features = self.feature_extractor(x)
        # Flatten the features
        features = torch.flatten(features, 1)
        # Apply the additional linear layer
        features = self.linear_layer(features)
        return features

# Example usage
# model = NeuralNetwork()
# model.to("cuda")
# input_tensor = torch.randn(1, 3, 32, 32).to("cuda")  # Example input tensor (batch_size, channels, height, width)
# # input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor (batch_size, channels, height, width)
# output_features = model(input_tensor)
# print(output_features.shape) 