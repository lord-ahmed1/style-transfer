import os
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image settings
IMAGE_SIZE = 512

# Path settings
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CONTENT_DIR = os.path.join(DATA_DIR, 'content')
STYLE_DIR = os.path.join(DATA_DIR, 'style')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# Model settings
CONTENT_LAYERS = ['conv_4', 'conv_5']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Training settings
NUM_STEPS = 1000
STYLE_WEIGHT = 3e8
CONTENT_WEIGHT = 10e1
STYLE_THRESHOLD = 0.7  # Adjustable style threshold for controlling transfer intensity

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
