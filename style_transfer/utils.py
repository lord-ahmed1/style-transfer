import torch
import torchvision.transforms as transforms
from PIL import Image

def image_loader(image_path, imsize, device):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0).to(device, torch.float)
    return image

def save_image(tensor, path):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(path)
