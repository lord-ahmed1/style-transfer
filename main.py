import torch
from utils import image_loader, save_image
from CNN_Model import VGG
from content_loss import ContentLoss
from style_loss import StyleLoss
from optimizer import run_optimization
import matplotlib.pyplot as plt
import numpy as np

# --- Settings ---
device = torch.device("cuda")
imsize = 512

content_img_path = "content2.jpg"
style_img_path = "Style2.jpg"
output_img_path = "Output.jpeg"

content_layers = ['conv_4', 'conv_5']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# --- Load Images ---
content_img = image_loader(content_img_path, imsize, device)
style_img = image_loader(style_img_path, imsize, device)
input_img = torch.randn(content_img.data.size(), device=device).clamp(0, 1).requires_grad_(True)

# --- Load VGG model ---
vgg = VGG(content_layers, style_layers).to(device).eval()

# --- Extract features ---
content_features, _ = vgg(content_img)
_, style_features = vgg(style_img)

# --- Loss modules ---
content_losses = [ContentLoss(f.detach()) for f in content_features]

style_losses=[]
for index,f in enumerate(style_features):
    layer_weight=(index+1)**5
    print(f'layer weight {layer_weight}')
    layer_style_loss=StyleLoss(f.detach(),layer_weight)
    style_losses.append(layer_style_loss)

style_intensity = 1


if style_intensity != 0:
    x = (1 - style_intensity) / style_intensity

    # --- Optimization ---
    output_img = run_optimization(
        vgg, input_img, content_losses, style_losses,
        num_steps=100, style_weight=3e8, content_weight= 50e3 * x
    )

else:
    output_img = content_img

img = output_img[0].cpu().detach().numpy()
img = img.transpose(1, 2, 0)

plt.imshow(img)


# --- Save Result ---
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


save_image(output_img, output_img_path)
print("Output image saved to", output_img_path)
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.cpu().clone()        # clone the tensor to not alter the original
    image = image.squeeze(0)            # remove the batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()

# Display the output image
imshow(output_img, title="Output")