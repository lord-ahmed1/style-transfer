import torch.nn as nn
import torch

def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        G = gram_matrix(x)
        return self.loss_fn(G, self.target)
