#!/usr/bin/env python3
"""
Demo script for neural style transfer.
This script provides a quick demonstration of the style transfer capabilities
when running in a container environment.
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Import project modules
from config.config import *
from utils.image_utils import image_loader, save_image, display_images, get_file_paths
from utils.optimizer import run_optimization
from models.model_factory import get_model
from losses.content_loss import ContentLoss
from losses.style_loss import StyleLoss
from losses.tv_loss import TotalVariationLoss


def run_demo(model_type="vgg", steps=300, threshold=0.7, save_dir="./demo_results"):
    """Run a simple demo of style transfer."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get available images
    content_paths, style_paths = get_file_paths(CONTENT_DIR, STYLE_DIR)
    if not content_paths or not style_paths:
        print("No content or style images found.")
        return
        
    # Use first content and style image
    content_path = content_paths[0]
    style_path = style_paths[0]
    
    print(f"Using content image: {os.path.basename(content_path)}")
    print(f"Using style image: {os.path.basename(style_path)}")
    
    # Load images
    content_img = image_loader(content_path, IMAGE_SIZE, DEVICE)
    style_img = image_loader(style_path, IMAGE_SIZE, DEVICE)
    
    # Initialize model
    model = get_model(model_type, content_layers=CONTENT_LAYERS,
                     style_layers=STYLE_LAYERS, device=DEVICE)
    
    # Create input image
    input_img = content_img.clone().requires_grad_(True)
    
    # Extract features
    content_features, _ = model(content_img)
    _, style_features = model(style_img)
    
    # Setup losses
    content_losses = [ContentLoss(f.detach()).to(DEVICE) for f in content_features]
    style_losses = []
    
    for idx, f in enumerate(style_features):
        layer_weight = (idx + 1)**2
        style_loss = StyleLoss(f.detach(), layer_weight, threshold).to(DEVICE)
        style_losses.append(style_loss)
    
    # Total variation loss
    tv_loss = TotalVariationLoss(weight=1.0).to(DEVICE)
    
    # Run optimization
    print(f"Running style transfer optimization for {steps} steps...")
    output_img, loss_history = run_optimization(
        model, input_img, content_losses, style_losses,
        tv_loss=tv_loss, num_steps=steps,
        style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT,
        tv_weight=1.0, style_threshold=threshold
    )
    
    # Create output filename
    content_name = Path(content_path).stem
    style_name = Path(style_path).stem
    output_path = os.path.join(save_dir, f"{content_name}_{style_name}_{model_type}.jpg")
    
    # Save result
    save_image(output_img, output_path)
    print(f"Output saved to {output_path}")
    
    # Create and save comparison
    fig = display_images(content_img, style_img, output_img)
    comparison_path = os.path.join(save_dir, f"{content_name}_{style_name}_comparison.jpg")
    plt.savefig(comparison_path)
    plt.close()
    print(f"Comparison image saved to {comparison_path}")
    
    # Plot and save loss history
    plt.figure(figsize=(10, 5))
    steps = range(1, len(loss_history['total'])+1)
    plt.plot(steps, loss_history['content'], label='Content Loss')
    plt.plot(steps, loss_history['style'], label='Style Loss')
    plt.plot(steps, loss_history['tv'], label='TV Loss')
    plt.plot(steps, loss_history['total'], label='Total Loss')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss Value')
    plt.title('Loss History')
    plt.legend()
    plt.yscale('log')
    loss_path = os.path.join(save_dir, f"{content_name}_{style_name}_loss.jpg")
    plt.savefig(loss_path)
    plt.close()
    print(f"Loss plot saved to {loss_path}")
    
    return output_img, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer Demo')
    parser.add_argument('--model', type=str, default='vgg', choices=['vgg', 'vit'],
                        help='Model type (vgg or vit)')
    parser.add_argument('--steps', type=int, default=300,
                        help='Number of optimization steps')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Style threshold (0-1)')
    parser.add_argument('--output', type=str, default='./demo_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if args.model == 'vit':
        try:
            import transformers
            print("Using Vision Transformer model")
        except ImportError:
            print("Warning: transformers library not found, falling back to VGG model")
            args.model = 'vgg'
    
    print(f"Demo configuration:")
    print(f"- Model: {args.model}")
    print(f"- Steps: {args.steps}")
    print(f"- Style threshold: {args.threshold}")
    print(f"- Output directory: {args.output}")
    print(f"- Device: {DEVICE}")
    
    run_demo(args.model, args.steps, args.threshold, args.output)
