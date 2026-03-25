import kagglehub

def download_nail_dataset():
    """Download dataset and return local path."""
    print("\n⬇️ Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nikhilgurav21/nail-disease-detection-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_nail_dataset()