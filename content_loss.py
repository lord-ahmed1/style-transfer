import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # Ensuring that the target doesn't require gradients
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.loss_fn(x, self.target)
