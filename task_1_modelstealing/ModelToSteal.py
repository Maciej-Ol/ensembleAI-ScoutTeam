import torchvision.models as models
import torch
from endpoints.model_stealing import model_stealing

class ModelToSteal:
    def get_embeddings(self, image):
        pass

class ModelToStealMockup(ModelToSteal):
    def __init__(self):
        self.arch = "resnet50"
        self.model = models.__dict__[self.arch]()
        checkpoint = torch.load("simsiam\checkpoint_0099.pth.tar", map_location="cpu")
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                # remove prefix
                state_dict[k[len("module.encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.fc = torch.nn.Identity()
    
    def get_embeddings(self, image):
        return self.model(image)

class ModelToStealOfficial(ModelToSteal):
    def get_embeddings(self, image):
        result = model_stealing(image)
        return result