# Neural Style Transfer Project

This project implements neural style transfer techniques based on the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al. The implementation includes both CNN-based and Vision Transformer-based approaches, with a dynamic style threshold parameter for controlling the intensity of style transfer.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new)

## Project Structure

```
/style-transfer/
├── config/             # Configuration settings
├── data/               # Data directory
│   ├── content/        # Content images
│   └── style/          # Style images  
├── losses/             # Loss function modules
├── models/             # Model implementations
│   ├── cnn/            # CNN-based models (VGG)
│   └── transformer/    # Vision Transformer models
├── notebooks/          # Jupyter notebooks for demos and analysis
├── results/            # Generated images and evaluations
└── utils/              # Utility functions
```
## Features

- Neural style transfer using pre-trained VGG19 model
- Style transfer with Vision Transformer (ViT) as a bonus implementation
- Dynamic control over style transfer intensity with a threshold parameter
- Batch processing capability for multiple content-style pairs
- Comprehensive loss tracking and visualization
- Command line interface for easy use

## Usage

### Command Line Interface

```bash
# Basic style transfer
python main.py --content data/content/your_content.jpg --style data/style/your_style.jpg

# Adjust style intensity
python main.py --content data/content/your_content.jpg --style data/style/your_style.jpg --threshold 0.5

# Use Vision Transformer instead of VGG
python main.py --content data/content/your_content.jpg --style data/style/your_style.jpg --model vit

# Batch processing of all images in the data directories
python main.py --batch
```

### Using the Jupyter Notebook

The project includes a Jupyter notebook in the `notebooks` directory that demonstrates the style transfer process and provides visualizations of the results.

```bash
cd notebooks
jupyter notebook style_transfer_demo.ipynb
```

## Model Architecture

### CNN-Based Model (VGG19)
- Uses pre-trained VGG19 model from torchvision
- Modified by replacing max pooling with average pooling
- Extracts features from specified content and style layers

### Vision Transformer Model (ViT)
- Uses pre-trained ViT model from the transformers library
- Adapts the transformer architecture for style transfer
- Extracts features from intermediate transformer layers

## Loss Functions

- **Content Loss**: Mean squared error between content features
- **Style Loss**: Mean squared error between Gram matrices of style features
- **Total Variation Loss**: Encourages spatial smoothness in the output

## Evaluation

The model's performance is evaluated based on:
1. Visual quality of the stylized images
2. Convergence behavior of the different loss components
3. Effect of the style threshold parameter on the output

## Examples

The `results` directory contains example outputs, including:
- Stylized images with different style thresholds
- Comparison images showing content, style, and output
- Loss plots showing the optimization process

## Docker and Codespaces Support

This project includes Docker and GitHub Codespaces configurations for easy setup and development.

### Running with Docker

1. Make sure you have Docker and Docker Compose installed
2. Clone the repository
3. Build and start the container:

```bash
docker-compose up --build
```

4. Access Jupyter Lab at http://localhost:8888
5. Run the demo script:

```bash
docker exec -it style-transfer-style-transfer-1 python demo.py
```

### Using GitHub Codespaces

1. Click the "Open in GitHub Codespaces" button at the top of this README
2. Wait for the environment to be set up (this may take a few minutes)
3. Once ready, you can:
   - Open the Jupyter notebooks in the `notebooks` directory
   - Run the demo script with `python demo.py`
   - Run style transfer with specific parameters: `python main.py --content data/content/Content.jpeg --style data/style/Style.jpeg --threshold 0.7`

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Set to control which GPUs are used
- `PYTHONPATH`: Automatically set to include the project root

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. arXiv:1508.06576.
- Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929.