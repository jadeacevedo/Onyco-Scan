import asyncio
import json
import os
import pathlib

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import display, clear_output, Image as IPImage
from PIL import Image
import ipywidgets as widgets

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# RAG dependencies
from sentence_transformers import SentenceTransformer
import faiss

plt.style.use('seaborn-v0_8-whitegrid')

print(f"TensorFlow Version: {tf.__version__}")
print("✅ Environment Ready")