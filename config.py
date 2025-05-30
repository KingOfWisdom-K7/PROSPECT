import os

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
EARLY_STOP_PATIENCE = 5
RANDOM_SEED = 42

# Data paths
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model saving directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Output directory for predictions and plots
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)


