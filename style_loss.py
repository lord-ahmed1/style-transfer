import torch.nn as nn
import torch

def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature,layer_weight=1):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_fn = nn.MSELoss()
        self.layer_weight = layer_weight

    def forward(self, x, weight=1.0):
        G = gram_matrix(x)
        loss = self.loss_fn(G, self.target)
        return loss * self.layer_weight
   
        