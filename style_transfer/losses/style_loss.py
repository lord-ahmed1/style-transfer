import torch
import torch.nn as nn

def gram_matrix(input_tensor):
    """
    Calculate the Gram matrix of the input tensor.
    
    The Gram matrix is used to capture style information (texture, color, etc.)
    by measuring feature correlations across spatial locations.
    
    Args:
        input_tensor: Feature map tensor with shape (batch, channel, height, width)
        
    Returns:
        Normalized Gram matrix
    """
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(c, h * w)
    G = torch.mm(features, features.t())  # Compute the inner product
    
    # Normalize by the size of the feature map
    return G / (c * h * w)

class StyleLoss(nn.Module):
    """
    Style loss module for neural style transfer.
    
    This module computes the Mean Squared Error between the
    Gram matrices of the style image and the generated image.
    
    Args:
        target_feature: Feature map from the style image
        layer_weight: Weight for this specific layer (deeper layers can have higher weights)
        style_threshold: Threshold parameter to control style transfer intensity (0-1)
    """
    def __init__(self, target_feature, layer_weight=1, style_threshold=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_fn = nn.MSELoss()
        self.layer_weight = layer_weight
        self.style_threshold = style_threshold

    def forward(self, x):
        G = gram_matrix(x)
        
        # Apply style threshold by interpolating between G and target
        if self.style_threshold < 1.0:
            # Linear interpolation between content gram matrix and style gram matrix
            G = G * (1 - self.style_threshold) + self.target * self.style_threshold
            
        loss = self.loss_fn(G, self.target)
        return loss * self.layer_weight
