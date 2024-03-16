import torchvision.models as models
import torch
from endpoints.requests2 import model_stealing
from torchvision import transforms
from  taskdataset import TaskDataset

from PIL import Image

class ModelToSteal:
    def get_embeddings(self, image, id=0):
        pass

class ModelToStealRandomMockup(ModelToSteal):
    def __init__(self):
        self.mapping = {}

    def get_embeddings(self, image: Image, id):
        if id not in self.mapping:
            self.mapping[id] = torch.rand(512)
        return self.mapping[id]

class ModelToStealMockup(ModelToSteal):
    def __init__(self):
        self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        # Define the path to the checkpoint file
        # checkpoint_path = './simsiam/checkpoint_0099.pth.tar'
        checkpoint_path = "task_1_modelstealing\simsiam\checkpoint_0099.pth.tar"

        # Load the pretrained ResNet-50 model
        self.model = models.resnet50(pretrained=False)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load the pretrained weights into the model
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 512)

        # Set the model to evaluation mode
        self.model.eval()


    def get_embeddings(self, image: Image):
        return self.model(self.transform(image).reshape(1, 3, 32, 32))

class ModelToStealOfficial(ModelToSteal):
    def __init__(self, max_noise = 1e-5):
        self.max_noise = max_noise
        self.iterations_to_denoise = 1

    def get_embeddings(self, image: Image, id = 0):
        image.save(f"task_1_modelstealing/data/images/{id}.png")
        result = model_stealing(f"task_1_modelstealing/data/images/{id}.png")
        return result
   
    def estimate_noise(self, original_img, original_vector):
        i = 1
        encoding = self.get_embeddings(original_img)
        dist = (original_vector - encoding).pow(2).sum().sqrt()
    
        while dist > self.max_noise:
            next_encoding = self.get_embeddings(original_img)
            encoding += next_encoding / i
            encoding = encoding * i / (i+1)
            dist = (original_vector - encoding).pow(2).sum().sqrt()
            i += 1

        self.iterations_to_denoise = i
        return i

    def get_denoised_embedding(self, image: Image, id = 0):
        encoding = 0
        for _ in range(self.iterations_to_denoise):
            encoding += torch.tensor(self.get_embeddings(image, id)) / self.iterations_to_denoise
            
        return encoding
    
if __name__ == "__main__":
    model = ModelToStealMockup()

    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1

    path_to_images = "task_1_modelstealing/data/ModelStealingPub.pt"

    dataset = torch.load(path_to_images)

    # Pass the tensor through the model
    output = model.model(dataset.imgs[0])

    # The output would be the model's prediction
    print(output.shape)