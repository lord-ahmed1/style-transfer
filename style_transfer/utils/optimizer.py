import torch
import torch.optim as optim
import copy

def run_optimization(model, input_img, content_losses, style_losses, 
                    tv_loss=None, num_steps=300, style_weight=1e6, 
                    content_weight=1, tv_weight=0, style_threshold=1.0):
    """
    Optimize the input image to match the content and style targets.
    
    Args:
        model: The feature extraction model (VGG or ViT)
        input_img: The initial image (usually a copy of the content image)
        content_losses: List of content loss modules
        style_losses: List of style loss modules
        tv_loss: Total variation loss module (optional)
        num_steps: Number of optimization steps
        style_weight: Weight of style loss
        content_weight: Weight of content loss
        tv_weight: Weight of total variation loss
        style_threshold: Style transfer intensity control (0-1)
        
    Returns:
        The optimized image
    """
    # Adjust style losses based on threshold
    for sl in style_losses:
        sl.style_threshold = style_threshold
    
    # Use LBFGS optimizer for better results
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    best_loss = float('inf')
    best_img = input_img.clone().detach()
    
    loss_history = {'content': [], 'style': [], 'tv': [], 'total': []}

    while run[0] <= num_steps:

        def closure():
            nonlocal best_loss, best_img
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()

            content_features, style_features = model(input_img)

            content_score = 0
            style_score = 0

            # Calculate content loss
            for cl, f in zip(content_losses, content_features):
                content_score += cl(f)

            # Calculate style loss
            for sl, f in zip(style_losses, style_features):
                style_score += sl(f)
                
            # Calculate total variation loss if provided
            tv_score = tv_loss(input_img) if tv_loss else 0
                
            # Calculate weighted total loss
            weighted_content_loss = content_weight * content_score
            weighted_style_loss = style_weight * style_score
            weighted_tv_loss = tv_weight * tv_score
            
            loss = weighted_style_loss + weighted_content_loss + weighted_tv_loss
            loss.backward()
            
            # Store loss history
            loss_history['content'].append(weighted_content_loss)
            loss_history['style'].append(weighted_style_loss)
            loss_history['tv'].append(weighted_tv_loss if isinstance(weighted_tv_loss, (int, float)) else weighted_tv_loss.item())
            loss_history['total'].append(loss.item())

            # Keep track of the best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_img = input_img.clone().detach()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {weighted_style_loss:.4f}, "
                      f"Content Loss: {weighted_content_loss:.4f}, "
                      f"TV Loss: {weighted_tv_loss if isinstance(weighted_tv_loss, (int, float)) else weighted_tv_loss.item():.4f}")

            return loss

        optimizer.step(closure)

    # Return the best image (with lowest loss)
    best_img.data.clamp_(0, 1)
    return best_img, loss_history
