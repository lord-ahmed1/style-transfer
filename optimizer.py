import torch.optim as optim
import torch

def run_optimization(model, input_img, content_losses, style_losses, num_steps=300, style_weight=1e6, content_weight=1):
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()], lr = 1)
    run = [0]
    best_loss = float('inf')
    best_img = input_img.clone().detach()

    while run[0] <= num_steps - 20:

        def closure():
            nonlocal best_loss, best_img
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()

            content_features, style_features = model(input_img)

            content_score = 0
            style_score = 0

            for cl, f in zip(content_losses, content_features):
                content_score += cl(f)

            for sl, f in zip(style_losses, style_features):
                style_score += sl(f)

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_img = input_img.clone().detach()

            run[0] += 1
            # if run[0] % 25 == 0:
                # print(f"Step {run[0]}: Style Loss: {style_score * 100000:.4f}, Content Loss: {content_score.item():.4f}")

            return loss
        
        optimizer.step(closure)

    best_img.data.clamp_(0, 1)
    return best_img
