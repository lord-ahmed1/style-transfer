import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to allow imports from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn.vgg_model import VGG
from models.transformer.vit_model import VisionTransformer

def get_model(model_type='vgg', content_layers=None, style_layers=None, device=torch.device('cuda')):
    """
    Factory function to create and return the appropriate model.
    
    Args:
        model_type: The type of model to use ('vgg' or 'vit')
        content_layers: List of layer names to extract content features from
        style_layers: List of layer names to extract style features from
        device: Device to place the model on
        
    Returns:
        The initialized model
    """
    if model_type.lower() == 'vgg':
        # Default layers for VGG if not specified
        if content_layers is None:
            content_layers = ['conv_4', 'conv_5']
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            
        model = VGG(content_layers, style_layers).to(device)
        
    elif model_type.lower() == 'vit':
        # Default layers for ViT are already set in the VisionTransformer class
        model = VisionTransformer(content_layers, style_layers).to(device)
        
    else:
        raise ValueError(f"Model type '{model_type}' not recognized. Choose 'vgg' or 'vit'.")
        
    # Set the model to evaluation mode
    model.eval()
    
    return model
