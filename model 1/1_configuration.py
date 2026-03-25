from system.imports.imports import tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

def system_setup():
    """Verifies GPU availability for Deep Learning."""
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU Detected: {len(gpus)} device(s). Training will be fast.")
    else:
        print("⚠️ No GPU detected. Training might be slow on CPU.")
