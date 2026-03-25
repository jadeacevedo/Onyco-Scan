from system.imports.imports import kagglehub, pathlib, tf


def download_data():
    """Downloads the specific Kaggle dataset."""
    print("\n⬇️ Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nikhilgurav21/nail-disease-detection-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    return path

def prepare_datasets(data_dir):
    """
    Creates training and validation datasets directly from the folder structure.
    The dataset contains classes: 'Acral Lentiginous Melanoma', 'Blue Finger',
    'Clubbing', 'Healthy Nail', 'Onychogryphosis', 'Pitting'
    """
    data_dir = pathlib.Path(data_dir)

    # Check for the 'images' or 'data' subdirectory if the root isn't the dataset directly
    # Some kaggle datasets nest files. We look for the folder containing class folders.
    if not (data_dir / 'Acral_Lentiginous_Melanoma').exists():
        # Try standard subfolder names often found in this specific dataset
        potential_dirs = list(data_dir.glob('*/'))
        for p in potential_dirs:
             if (p / 'Acral_Lentiginous_Melanoma').exists():
                 data_dir = p
                 break

    print(f"📂 Loading images from: {data_dir}")

    # Data Augmentation: Crucial for medical images to learn invariance
    # (e.g., a vertical band should be detected even if the finger is slightly rotated)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Mimic different finger angles
        width_shift_range=0.2,  # Shift band position
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,         # Zoom in on nail texture
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2    # 20% for testing/validation
    )

    print("⚙️ Generating Data Pipeline...")

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, val_generator