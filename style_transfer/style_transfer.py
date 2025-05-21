import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add the project root directory to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import project modules
from config.config import *
from utils.image_utils import image_loader, save_image, display_images, get_file_paths
from utils.data_utils import create_dataloaders
from utils.optimizer import run_optimization
from models.model_factory import get_model
from losses.content_loss import ContentLoss
from losses.style_loss import StyleLoss
from losses.tv_loss import TotalVariationLoss


def style_transfer(content_path, style_path, output_path, model_type='vgg', 
                  style_threshold=0.7, num_steps=1000, style_weight=1e6, 
                  content_weight=1, tv_weight=1, device=DEVICE):
    """
    Perform neural style transfer on a single image pair.
    
    Args:
        content_path: Path to the content image
        style_path: Path to the style image
        output_path: Path to save the result
        model_type: 'vgg' or 'vit'
        style_threshold: Style transfer intensity (0-1)
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        tv_weight: Weight for total variation loss
        device: Computing device
        
    Returns:
        The stylized image and loss history
    """
    # Load images
    content_img = image_loader(content_path, IMAGE_SIZE, device)
    style_img = image_loader(style_path, IMAGE_SIZE, device)
    input_img = content_img.clone().requires_grad_(True)
    
    # Initialize the model
    model = get_model(model_type, content_layers=CONTENT_LAYERS, 
                     style_layers=STYLE_LAYERS, device=device)
    
    # Extract features
    content_features, _ = model(content_img)
    _, style_features = model(style_img)
    
    # Setup content loss modules
    content_losses = [ContentLoss(f.detach()).to(device) for f in content_features]
    
    # Setup style loss modules with layer weighting
    style_losses = []
    for idx, f in enumerate(style_features):
        # Use exponential weighting based on layer depth
        layer_weight = (idx + 1)**2  # Lighter weighting than original (idx + 1)**5
        style_loss = StyleLoss(f.detach(), layer_weight, style_threshold).to(device)
        style_losses.append(style_loss)
    
    # Setup total variation loss
    tv_loss = TotalVariationLoss(weight=1).to(device) if tv_weight > 0 else None
    
    # Run optimization
    output_img, loss_history = run_optimization(
        model, input_img, content_losses, style_losses,
        tv_loss=tv_loss, num_steps=num_steps, 
        style_weight=style_weight, content_weight=content_weight,
        tv_weight=tv_weight, style_threshold=style_threshold
    )
    
    # Save result
    save_image(output_img, output_path)
    print(f"Output image saved to {output_path}")
    
    return output_img, loss_history


def batch_style_transfer(model_type='vgg', style_threshold=0.7, batch_size=1):
    """
    Perform style transfer on multiple content-style pairs using a data loader.
    
    Args:
        model_type: 'vgg' or 'vit'
        style_threshold: Style transfer intensity (0-1)
        batch_size: Batch size for the data loader
    """
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        CONTENT_DIR, STYLE_DIR, 
        batch_size=batch_size, 
        image_size=IMAGE_SIZE
    )
    
    # Initialize the model
    model = get_model(model_type, content_layers=CONTENT_LAYERS, 
                     style_layers=STYLE_LAYERS, device=DEVICE)
    
    # Process training examples
    print(f"Processing {len(train_loader)} training examples...")
    for idx, batch in enumerate(tqdm(train_loader)):
        content_img = batch['content'].to(DEVICE)
        style_img = batch['style'].to(DEVICE)
        
        # Base output filename on content and style image names
        content_name = os.path.basename(batch['content_path'][0]).split('.')[0]
        style_name = os.path.basename(batch['style_path'][0]).split('.')[0]
        output_name = f"{content_name}_{style_name}_{style_threshold}.jpg"
        output_path = os.path.join(RESULTS_DIR, output_name)
        
        # Clone content image as starting point
        input_img = content_img.clone().requires_grad_(True)
        
        # Extract features
        content_features, _ = model(content_img)
        _, style_features = model(style_img)
        
        # Setup loss modules
        content_losses = [ContentLoss(f.detach()).to(DEVICE) for f in content_features]
        
        style_losses = []
        for i, f in enumerate(style_features):
            layer_weight = (i + 1)**2
            style_loss = StyleLoss(f.detach(), layer_weight, style_threshold).to(DEVICE)
            style_losses.append(style_loss)
        
        # Total variation loss
        tv_loss = TotalVariationLoss(weight=1).to(DEVICE)
        
        # Run optimization
        output_img, _ = run_optimization(
            model, input_img, content_losses, style_losses,
            tv_loss=tv_loss, num_steps=NUM_STEPS, 
            style_weight=STYLE_WEIGHT, content_weight=CONTENT_WEIGHT,
            tv_weight=1.0, style_threshold=style_threshold
        )
        
        # Save result
        save_image(output_img, output_path)
        print(f"Saved output to {output_path}")


def main():
    """Main function to parse arguments and run style transfer."""
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, default=None, help='Path to content image')
    parser.add_argument('--style', type=str, default=None, help='Path to style image')
    parser.add_argument('--output', type=str, default=None, help='Path to output image')
    parser.add_argument('--model', type=str, default='vgg', choices=['vgg', 'vit'], help='Model type')
    parser.add_argument('--threshold', type=float, default=0.7, help='Style threshold (0-1)')
    parser.add_argument('--steps', type=int, default=NUM_STEPS, help='Number of optimization steps')
    parser.add_argument('--style_weight', type=float, default=STYLE_WEIGHT, help='Style loss weight')
    parser.add_argument('--content_weight', type=float, default=CONTENT_WEIGHT, help='Content loss weight')
    parser.add_argument('--tv_weight', type=float, default=1.0, help='Total variation loss weight')
    parser.add_argument('--batch', action='store_true', help='Process all images in batch mode')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for batch processing')
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.batch:
        batch_style_transfer(
            model_type=args.model,
            style_threshold=args.threshold,
            batch_size=args.batch_size
        )
        return
    
    # Single image processing mode
    if not args.content or not args.style:
        # Use default images if none provided
        args.content = os.path.join(CONTENT_DIR, os.listdir(CONTENT_DIR)[0])
        args.style = os.path.join(STYLE_DIR, os.listdir(STYLE_DIR)[0])
        print(f"Using default content: {args.content}")
        print(f"Using default style: {args.style}")
        
    # Default output path if not provided
    if not args.output:
        content_name = os.path.basename(args.content).split('.')[0]
        style_name = os.path.basename(args.style).split('.')[0]
        args.output = os.path.join(RESULTS_DIR, f"{content_name}_{style_name}_{args.threshold}.jpg")
    
    # Run style transfer
    output_img, loss_history = style_transfer(
        args.content, args.style, args.output,
        model_type=args.model, style_threshold=args.threshold,
        num_steps=args.steps, style_weight=args.style_weight,
        content_weight=args.content_weight, tv_weight=args.tv_weight
    )
    
    # Display results
    content_img = image_loader(args.content, IMAGE_SIZE, DEVICE)
    style_img = image_loader(args.style, IMAGE_SIZE, DEVICE)
    fig = display_images(content_img, style_img, output_img)
    plt.savefig(os.path.join(RESULTS_DIR, f"{content_name}_{style_name}_comparison.jpg"))
    
    # Plot loss history
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
    plt.savefig(os.path.join(RESULTS_DIR, f"{content_name}_{style_name}_loss.jpg"))


if __name__ == "__main__":
    main()
