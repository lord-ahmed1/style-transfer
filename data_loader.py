import os
import random
import torch
from PIL import Image
from torchvision import transforms


def load_image(path, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(path).convert('RGB')
    image = loader(image).unsqueeze(0).to(device)
    return image


def generate_style_content_pairs(content_dir, style_dir, imsize, device):
    content_images = os.listdir(content_dir)
    style_images = os.listdir(style_dir)

    random.shuffle(content_images)  # ensure unique and shuffled order

    for content_name in content_images:
        style_name = random.choice(style_images)

        content_img_path = os.path.join(content_dir, content_name)
        style_img_path = os.path.join(style_dir, style_name)

        content_img = load_image(content_img_path, imsize, device)
        style_img = load_image(style_img_path, imsize, device)
        style_intensity = random.uniform(0.1, 0.9)

        yield content_img, style_img, style_intensity
