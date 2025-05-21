#!/usr/bin/env python3
"""
Evaluation script for neural style transfer models.

This script evaluates both CNN and Vision Transformer models on a set of
content-style pairs with various style thresholds.
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import project modules
from config.config import *
from utils.image_utils import image_loader, save_image, get_file_paths
from utils.data_utils import create_dataloaders
from models.model_factory import get_model
from losses.content_loss import ContentLoss
from losses.style_loss import StyleLoss
from losses.tv_loss import TotalVariationLoss
from utils.optimizer import run_optimization


def evaluate_model(model_type, content_paths, style_paths, thresholds=[0.2, 0.4, 0.6, 0.8, 1.0],
                  num_steps=300, style_weight=1e6, content_weight=1, tv_weight=1):
    """
    Evaluate the model on multiple content-style pairs with different thresholds.
    
    Args:
        model_type: 'vgg' or 'vit'
        content_paths: List of content image paths
        style_paths: List of style image paths
        thresholds: List of style thresholds to evaluate
        num_steps: Number of optimization steps
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        tv_weight: Weight for total variation loss
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    # Use at most 5 content and 5 style images for evaluation to keep it reasonable
    content_paths = content_paths[:5]
    style_paths = style_paths[:5]
    
    # Initialize the model
    model = get_model(model_type, content_layers=CONTENT_LAYERS, 
                     style_layers=STYLE_LAYERS, device=DEVICE)
    
    # Create an evaluation directory
    eval_dir = os.path.join(RESULTS_DIR, f"evaluation_{model_type}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Iterate over content-style pairs
    for i, content_path in enumerate(content_paths):
        # Use one style image per content image for simplicity
        style_path = style_paths[i % len(style_paths)]
        
        content_name = os.path.basename(content_path).split('.')[0]
        style_name = os.path.basename(style_path).split('.')[0]
        
        # Load images
        content_img = image_loader(content_path, IMAGE_SIZE, DEVICE)
        style_img = image_loader(style_path, IMAGE_SIZE, DEVICE)
        
        print(f"\nProcessing content: {content_name}, style: {style_name}")
        
        # Create a subdirectory for this content-style pair
        pair_dir = os.path.join(eval_dir, f"{content_name}_{style_name}")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Extract features (only need to do this once per content-style pair)
        content_features, _ = model(content_img)
        _, style_features = model(style_img)
        
        # Setup content loss modules
        content_losses = [ContentLoss(f.detach()).to(DEVICE) for f in content_features]
        
        # Iterate over thresholds
        for threshold in thresholds:
            print(f"  Threshold: {threshold}")
            
            # Clone content image as starting point
            input_img = content_img.clone().requires_grad_(True)
            
            # Setup style loss modules with layer weighting
            style_losses = []
            for idx, f in enumerate(style_features):
                layer_weight = (idx + 1)**2
                style_loss = StyleLoss(f.detach(), layer_weight, threshold).to(DEVICE)
                style_losses.append(style_loss)
            
            # Setup total variation loss
            tv_loss = TotalVariationLoss(weight=1).to(DEVICE) if tv_weight > 0 else None
            
            # Run optimization
            output_img, loss_history = run_optimization(
                model, input_img, content_losses, style_losses,
                tv_loss=tv_loss, num_steps=num_steps,
                style_weight=style_weight, content_weight=content_weight,
                tv_weight=tv_weight, style_threshold=threshold
            )
            
            # Save the stylized image
            output_path = os.path.join(pair_dir, f"threshold_{threshold:.1f}.jpg")
            save_image(output_img, output_path)
            
            # Save loss plot
            plt.figure(figsize=(10, 5))
            steps = range(1, len(loss_history['total'])+1)
            plt.plot(steps, loss_history['content'], label='Content Loss')
            plt.plot(steps, loss_history['style'], label='Style Loss')
            plt.plot(steps, loss_history['tv'], label='TV Loss')
            plt.plot(steps, loss_history['total'], label='Total Loss')
            plt.xlabel('Optimization Steps')
            plt.ylabel('Loss Value')
            plt.title(f'Loss History (Threshold = {threshold:.1f})')
            plt.legend()
            plt.yscale('log')
            plt.savefig(os.path.join(pair_dir, f"loss_threshold_{threshold:.1f}.jpg"))
            plt.close()
            
            # Record results
            results.append({
                'model_type': model_type,
                'content_image': content_name,
                'style_image': style_name,
                'threshold': threshold,
                'final_content_loss': loss_history['content'][-1],
                'final_style_loss': loss_history['style'][-1],
                'final_tv_loss': loss_history['tv'][-1],
                'final_total_loss': loss_history['total'][-1],
                'output_path': output_path
            })
    
    # Create a comparison grid for each content-style pair
    for content_path in content_paths:
        for style_path in style_paths:
            content_name = os.path.basename(content_path).split('.')[0]
            style_name = os.path.basename(style_path).split('.')[0]
            pair_dir = os.path.join(eval_dir, f"{content_name}_{style_name}")
            
            if not os.path.exists(pair_dir):
                continue
                
            # Create a grid of all threshold results
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Load content and style images
            content_img = Image.open(content_path).convert('RGB')
            style_img = Image.open(style_path).convert('RGB')
            
            # Display content and style images
            axes[0, 0].imshow(content_img)
            axes[0, 0].set_title('Content Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(style_img)
            axes[0, 1].set_title('Style Image')
            axes[0, 1].axis('off')
            
            # Hide one subplot
            axes[0, 2].axis('off')
            
            # Display results for different thresholds
            for i, threshold in enumerate(thresholds[:5]):  # Show up to 5 thresholds
                output_path = os.path.join(pair_dir, f"threshold_{threshold:.1f}.jpg")
                if os.path.exists(output_path):
                    output_img = Image.open(output_path).convert('RGB')
                    
                    row = 1 if i >= 3 else 0
                    col = i % 3 if i < 3 else i - 3
                    
                    if row == 0 and col == 2:
                        # Skip the reserved spot
                        continue
                        
                    axes[row, col].imshow(output_img)
                    axes[row, col].set_title(f'Threshold = {threshold:.1f}')
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(pair_dir, "threshold_comparison.jpg"))
            plt.close()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv_path = os.path.join(eval_dir, f"{model_type}_evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    return results_df


def compare_models(content_paths, style_paths, thresholds=[0.6]):
    """
    Compare VGG and ViT models on the same content-style pairs.
    
    Args:
        content_paths: List of content image paths
        style_paths: List of style image paths
        thresholds: List of style thresholds to evaluate
        
    Returns:
        DataFrame with comparison results
    """
    # Use at most 3 content and 3 style images for comparison
    content_paths = content_paths[:3]
    style_paths = style_paths[:3]
    
    # Create comparison directory
    comparison_dir = os.path.join(RESULTS_DIR, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Initialize both models
    vgg_model = get_model('vgg', content_layers=CONTENT_LAYERS, style_layers=STYLE_LAYERS, device=DEVICE)
    vit_model = get_model('vit', device=DEVICE)
    
    results = []
    
    # Iterate over content-style pairs
    for content_path in content_paths:
        for style_path in style_paths:
            content_name = os.path.basename(content_path).split('.')[0]
            style_name = os.path.basename(style_path).split('.')[0]
            
            print(f"\nComparing models on content: {content_name}, style: {style_name}")
            
            # Create a subdirectory for this content-style pair
            pair_dir = os.path.join(comparison_dir, f"{content_name}_{style_name}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # Load images
            content_img = image_loader(content_path, IMAGE_SIZE, DEVICE)
            style_img = image_loader(style_path, IMAGE_SIZE, DEVICE)
            
            # Iterate over thresholds
            for threshold in thresholds:
                print(f"  Threshold: {threshold}")
                
                # Process with VGG
                print("    Processing with VGG...")
                vgg_content_features, _ = vgg_model(content_img)
                _, vgg_style_features = vgg_model(style_img)
                
                vgg_content_losses = [ContentLoss(f.detach()).to(DEVICE) for f in vgg_content_features]
                
                vgg_style_losses = []
                for idx, f in enumerate(vgg_style_features):
                    layer_weight = (idx + 1)**2
                    style_loss = StyleLoss(f.detach(), layer_weight, threshold).to(DEVICE)
                    vgg_style_losses.append(style_loss)
                
                vgg_tv_loss = TotalVariationLoss(weight=1).to(DEVICE)
                
                vgg_input_img = content_img.clone().requires_grad_(True)
                vgg_output_img, vgg_loss_history = run_optimization(
                    vgg_model, vgg_input_img, vgg_content_losses, vgg_style_losses,
                    tv_loss=vgg_tv_loss, num_steps=300,
                    style_weight=1e6, content_weight=1, tv_weight=1, style_threshold=threshold
                )
                
                vgg_output_path = os.path.join(pair_dir, f"vgg_threshold_{threshold:.1f}.jpg")
                save_image(vgg_output_img, vgg_output_path)
                
                # Process with ViT
                print("    Processing with ViT...")
                vit_content_features, _ = vit_model(content_img)
                _, vit_style_features = vit_model(style_img)
                
                vit_content_losses = [ContentLoss(f.detach()).to(DEVICE) for f in vit_content_features]
                
                vit_style_losses = []
                for idx, f in enumerate(vit_style_features):
                    layer_weight = (idx + 1)**2
                    style_loss = StyleLoss(f.detach(), layer_weight, threshold).to(DEVICE)
                    vit_style_losses.append(style_loss)
                
                vit_tv_loss = TotalVariationLoss(weight=1).to(DEVICE)
                
                vit_input_img = content_img.clone().requires_grad_(True)
                vit_output_img, vit_loss_history = run_optimization(
                    vit_model, vit_input_img, vit_content_losses, vit_style_losses,
                    tv_loss=vit_tv_loss, num_steps=300,
                    style_weight=1e5, content_weight=10, tv_weight=1, style_threshold=threshold
                )
                
                vit_output_path = os.path.join(pair_dir, f"vit_threshold_{threshold:.1f}.jpg")
                save_image(vit_output_img, vit_output_path)
                
                # Create comparison figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                # Convert tensor to numpy for display
                def tensor_to_np_img(tensor):
                    img = tensor.cpu().clone().detach().numpy().squeeze(0)
                    img = img.transpose(1, 2, 0)
                    img = np.clip(img, 0, 1)
                    return img
                
                axes[0, 0].imshow(tensor_to_np_img(content_img))
                axes[0, 0].set_title('Content Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(tensor_to_np_img(style_img))
                axes[0, 1].set_title('Style Image')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(tensor_to_np_img(vgg_output_img))
                axes[1, 0].set_title('VGG Output')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(tensor_to_np_img(vit_output_img))
                axes[1, 1].set_title('ViT Output')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(pair_dir, f"comparison_threshold_{threshold:.1f}.jpg"))
                plt.close()
                
                # Record results
                results.append({
                    'content_image': content_name,
                    'style_image': style_name,
                    'threshold': threshold,
                    'vgg_content_loss': vgg_loss_history['content'][-1],
                    'vgg_style_loss': vgg_loss_history['style'][-1],
                    'vgg_total_loss': vgg_loss_history['total'][-1],
                    'vit_content_loss': vit_loss_history['content'][-1],
                    'vit_style_loss': vit_loss_history['style'][-1],
                    'vit_total_loss': vit_loss_history['total'][-1],
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv_path = os.path.join(comparison_dir, "model_comparison_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    
    return results_df


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Neural Style Transfer')
    parser.add_argument('--model', type=str, default='both', choices=['vgg', 'vit', 'both'], 
                        help='Which model to evaluate')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8, 1.0], 
                        help='Style thresholds to evaluate')
    parser.add_argument('--steps', type=int, default=300, 
                        help='Number of optimization steps')
    
    args = parser.parse_args()
    
    # Get content and style paths
    content_paths, style_paths = get_file_paths(CONTENT_DIR, STYLE_DIR)
    
    if not content_paths or not style_paths:
        print("Error: No content or style images found.")
        return
    
    print(f"Found {len(content_paths)} content images and {len(style_paths)} style images.")
    
    if args.model == 'vgg' or args.model == 'both':
        print("\nEvaluating VGG model:")
        vgg_results = evaluate_model(
            'vgg', content_paths, style_paths,
            thresholds=args.thresholds, num_steps=args.steps
        )
    
    if args.model == 'vit' or args.model == 'both':
        try:
            import transformers
            print("\nEvaluating ViT model:")
            vit_results = evaluate_model(
                'vit', content_paths, style_paths,
                thresholds=args.thresholds, num_steps=args.steps,
                style_weight=1e5, content_weight=10
            )
        except ImportError:
            print("\nSkipping ViT evaluation: transformers library not found.")
    
    if args.model == 'both':
        try:
            import transformers
            print("\nComparing VGG and ViT models:")
            comparison_results = compare_models(
                content_paths, style_paths,
                thresholds=[0.6]  # Use one threshold for comparison
            )
        except ImportError:
            print("\nSkipping model comparison: transformers library not found.")


if __name__ == "__main__":
    main()
