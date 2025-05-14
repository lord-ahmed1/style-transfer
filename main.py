import torch
from utils import image_loader, save_image
from CNN_Model import VGG
from content_loss import ContentLoss
from style_loss import StyleLoss
from optimizer import run_optimization

# --- Settings ---
device = torch.device("cuda")
imsize = 512

content_img_path = "Content2.jpg"
style_img_path = "Style2.jpg"
output_img_path = "Output.jpeg"

content_layers = ['conv_4', 'conv_5']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# --- Load Images ---
content_img = image_loader(content_img_path, imsize, device)
style_img = image_loader(style_img_path, imsize, device)
input_img = content_img.clone().requires_grad_(True)

# --- Load VGG model ---
vgg = VGG(content_layers, style_layers).to(device).eval()

# --- Extract features ---
content_features, _ = vgg(content_img)
_, style_features = vgg(style_img)

# --- Loss modules ---
content_losses = [ContentLoss(f.detach()) for f in content_features]
style_losses = [StyleLoss(f.detach()) for f in style_features]

# --- Optimization ---
output_img = run_optimization(
    vgg, input_img, content_losses, style_losses,
    num_steps=1000, style_weight=1e8, content_weight=1
)

# --- Save Result ---
save_image(output_img, output_img_path)
print("Output image saved to", output_img_path)
