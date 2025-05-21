import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def image_loader(image_path, imsize, device):
    """Load an image and convert it to a torch tensor."""
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0).to(device, torch.float)
    return image


def save_image(tensor, path):
    """Save a tensor as an image."""
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(path)


def display_images(content_img, style_img, output_img, figsize=(15, 5)):
    """Display the content, style, and output images."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Convert tensors to numpy arrays for display
    def tensor_to_np_img(tensor):
        img = tensor.cpu().clone().detach().numpy().squeeze(0)
        img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        return img
    
    # Display images
    ax1.imshow(tensor_to_np_img(content_img))
    ax1.set_title('Content Image')
    ax1.axis('off')
    
    ax2.imshow(tensor_to_np_img(style_img))
    ax2.set_title('Style Image')
    ax2.axis('off')
    
    ax3.imshow(tensor_to_np_img(output_img))
    ax3.set_title('Generated Image')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig


def get_file_paths(content_dir, style_dir):
    """Get all content and style image file paths."""
    content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return content_paths, style_paths
