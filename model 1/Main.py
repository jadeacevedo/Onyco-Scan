
import sys
import pathlib
import importlib.util

# Ensure we can import from system/imports path regardless of working directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # /workspaces/Onyco-Scan
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system.imports.imports import *
from system.imports import dataset as system_dataset


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

MODEL_DIR = pathlib.Path(__file__).resolve().parent
config_module = import_module_from_path("m1_config", MODEL_DIR / "1_configuration.py")
data_module = import_module_from_path("m1_data", MODEL_DIR / "2_datasetacquisition.py")
transfer_module = import_module_from_path("m1_transfer", MODEL_DIR / "3_TransferLearning.py")
training_module = import_module_from_path("m1_training", MODEL_DIR / "4_Training&Eval.py")

# Propagate configuration constants into module namespaces
data_module.IMG_SIZE = config_module.IMG_SIZE
data_module.BATCH_SIZE = config_module.BATCH_SIZE
transfer_module.LEARNING_RATE = config_module.LEARNING_RATE
training_module.EPOCHS = config_module.EPOCHS


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # System setup from model module
    config_module.system_setup()

    # 1. Download
    dataset_path = data_module.download_data()

    # 2. Prepare
    train_gen, val_gen = data_module.prepare_datasets(dataset_path)
    num_classes = len(train_gen.class_indices)
    print(f"ℹ️ Classes Detected: {list(train_gen.class_indices.keys())}")

    # 3. Build
    model = transfer_module.build_model(num_classes)

    # 4. Train
    model, history, final_acc = training_module.train_and_evaluate(model, train_gen, val_gen)

    # 5. Report
    training_module.generate_report(model, val_gen)

    # 6. Save
    save_path = "nail_melanoma_model.h5"
    model.save(save_path)
    print(f"\n💾 Model saved to {save_path}")

    # 7. Specific Melanoma Check Logic
    print("\n--- Diagnosis Logic ---")
    print("The system is trained to detect 'Acral Lentiginous Melanoma'.")
    print("If this class is predicted, verify 'Hutchinson sign' (pigment on skin) and vertical band width >3mm manually.")