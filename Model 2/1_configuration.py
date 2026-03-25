import os
import tensorflow as tf
from tensorflow.keras import layers

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
TRAIN_DIR = "/kaggle/input/nail-disease-detection-dataset/data/train"
VAL_DIR = "/kaggle/input/nail-disease-detection-dataset/data/validation"

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])


def get_dataset_paths(kaggle_base_path=None):
    """Resolve train/val directories for dataset pipeline."""
    if kaggle_base_path:
        train = os.path.join(kaggle_base_path, "data", "train")
        val = os.path.join(kaggle_base_path, "data", "validation")
        return train, val
    return TRAIN_DIR, VAL_DIR

