import torch
import torch.optim as optim
from CNN_Model import VGG
from content_loss import ContentLoss
from style_loss import StyleLoss
from optimizer import run_optimization
from data_loader import generate_style_content_pairs

# --- Settings ---
device = torch.device("cuda")
imsize = 512
content_layers = ['conv_4', 'conv_5']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
content_dir = "content"
style_dir = "style"

# --- Initialize VGG model and optimizer ---
vgg = VGG(content_layers, style_layers).to(device)
optimizer = optim.Adam(vgg.parameters(), lr=1e-5)

# --- Fine-tuning loop ---
pair_generator = generate_style_content_pairs(content_dir, style_dir, imsize, device)

for i, (content_img, style_img, style_intensity) in enumerate(pair_generator):
    input_img = torch.randn_like(content_img).clamp(0, 1).requires_grad_(True)

    # --- Compute intensity scaling factor ---
    x = (1 - style_intensity) / style_intensity

    # --- Extract target features ---
    with torch.no_grad():
        content_feats_target, _ = vgg(content_img)
        _, style_feats_target = vgg(style_img)

    # --- Loss modules with target features ---
    content_losses = [ContentLoss(f.detach()) for f in content_feats_target]
    style_losses = [StyleLoss(f.detach(), (j + 1) ** 5) for j, f in enumerate(style_feats_target)]

    # --- Optimize the image (this does NOT update VGG) ---
    output_img = run_optimization(
        vgg, input_img, content_losses, style_losses,
        num_steps=30,
        style_weight=1e6, content_weight=x*10**3.5
    )

    # --- Forward pass with the stylized image ---
    stylized_content_feats, stylized_style_feats = vgg(output_img)

    # --- Compute final losses (to backpropagate to VGG) ---
    content_loss = sum([cl(f) for cl, f in zip(content_losses, stylized_content_feats)])
    style_loss = sum([sl(f) for sl, f in zip(style_losses, stylized_style_feats)])
    loss = content_loss * 1 + style_loss * 1e6

    # --- Backprop to update VGG ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(f"[{i+1}/9000] Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item()*10000:.4f}")
    print(f"[{i+1}/9000] total Loss: {loss.item():.4f} Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item()*10000:.4f}")
