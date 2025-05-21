import torch.nn as nn

class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for promoting spatial smoothness in the generated image.
    
    This loss encourages spatial smoothness in the output image by penalizing
    large differences between adjacent pixel values.
    """
    def __init__(self, weight=1):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
        
    def forward(self, img):
        # Calculate the differences in the x direction
        h_tv = ((img[:, :, 1:, :] - img[:, :, :-1, :])**2).sum()
        # Calculate the differences in the y direction
        w_tv = ((img[:, :, :, 1:] - img[:, :, :, :-1])**2).sum()
        
        return self.weight * (h_tv + w_tv)
