import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionTransformer(nn.Module):
    """
    Pre-trained Vision Transformer (ViT) model for neural style transfer.
    
    This class loads a pre-trained ViT model and modifies it to extract
    features from intermediate layers for content and style representation.
    """
    def __init__(self, content_layers=None, style_layers=None, pretrained=True):
        super(VisionTransformer, self).__init__()
        
        # Default layers to extract features from if not specified
        self.content_layers = content_layers if content_layers else ['layer.9', 'layer.11']
        self.style_layers = style_layers if style_layers else ['layer.0', 'layer.2', 'layer.5', 'layer.8', 'layer.11']
        
        # Load pre-trained ViT model
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
            
        # Freeze the model parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # Resize input to 224x224 if needed (ViT's expected input size)
        batch_size, channels, height, width = x.shape
        if height != 224 or width != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        # Store the content and style features
        content_feats = []
        style_feats = []
        
        # Get all hidden states from the transformer
        outputs = self.vit(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Extract features from specified layers
        for i, layer_output in enumerate(hidden_states):
            layer_name = f'layer.{i}'
            
            # Reshape from [batch_size, sequence_length, hidden_size] to [batch_size, hidden_size, height, width]
            # Skip the CLS token (first token)
            feature_map = layer_output[:, 1:, :].reshape(batch_size, 14, 14, -1).permute(0, 3, 1, 2)
            
            if layer_name in self.content_layers:
                content_feats.append(feature_map)
            if layer_name in self.style_layers:
                style_feats.append(feature_map)
        
        return content_feats, style_feats
