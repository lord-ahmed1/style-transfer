import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class StyleTransferDataset(Dataset):
    """Dataset for neural style transfer."""
    def __init__(self, content_dir, style_dir, transform=None, size=512):
        self.content_paths = [os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.style_paths = [os.path.join(style_dir, f) for f in os.listdir(style_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.content_paths)
    
    def __getitem__(self, idx):
        # For simplicity, we'll use modulo to cycle through style images if there are fewer styles than contents
        content_path = self.content_paths[idx]
        style_path = self.style_paths[idx % len(self.style_paths)]
        
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")
        
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        
        return {
            'content': content_img,
            'style': style_img,
            'content_path': content_path,
            'style_path': style_path
        }


def create_dataloaders(content_dir, style_dir, batch_size=1, image_size=512, train_split=0.8):
    """Create training and validation dataloaders."""
    dataset = StyleTransferDataset(content_dir, style_dir, size=image_size)
    
    # Split dataset into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader
