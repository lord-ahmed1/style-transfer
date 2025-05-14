import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(VGG, self).__init__()
        
        # Store the content and style layers
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # Load pre-trained VGG19 model from torchvision
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Initialize lists to store layers and their corresponding names
        self.layers = []
        self.layer_names = []
        conv_count = 0

        # Iterate over layers in the VGG model
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                name = f"conv_{conv_count}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{conv_count}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{conv_count}"
                layer = nn.AvgPool2d(kernel_size=2, stride=2) 
            self.layers.append(layer)
            self.layer_names.append(name)

        # Create the sequential model with the selected layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # Store the content and style features
        content_feats = []
        style_feats = []
        
        # Pass input through each layer of the VGG model
        for layer, name in zip(self.model, self.layer_names):
            x = layer(x)
            if name in self.content_layers:
                content_feats.append(x)
            if name in self.style_layers:
                style_feats.append(x)
        
        return content_feats, style_feats
