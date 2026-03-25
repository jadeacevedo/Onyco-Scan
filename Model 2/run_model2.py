import importlib.util
import pathlib
import sys

from system.imports.dataset import download_nail_dataset

WORKDIR = pathlib.Path(__file__).resolve().parent


def _load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training_pipeline():
    config = _load_module('m2_config', WORKDIR / '1_configuration.py')
    data_module = _load_module('m2_data', WORKDIR / '2_DatasetaAcquisition.py')
    train_module = _load_module('m2_train', WORKDIR / '3_Training&Eval.py')
    rag_module = _load_module('m2_rag', WORKDIR / '4_RAG.py')

    print('\n=== Step 0: Ensure dataset is downloaded ===')
    try:
        data_path = download_nail_dataset()
    except Exception as e:
        print('⚠️ Could not download dataset via system/imports/dataset.py:', e)
        print('⚠️ Assuming local dataset is already present.')
        data_path = None

    train_dir, val_dir = config.get_dataset_paths(data_path)

    print('\n=== Step 1: Load and map data ===')
    loader = data_module.DataLoader(train_dir=train_dir, val_dir=val_dir, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, seed=config.SEED)
    train_ds, val_ds, class_names = loader.load_data()
    print('Loaded data sets', class_names)

    print('\n=== Step 2: Build model ===')
    model, base_model = train_module.build_model(img_size=config.IMG_SIZE, num_classes=len(class_names))

    print('\n=== Step 3: Compute class weights ===')
    class_weights = train_module.compute_class_weights(train_ds)
    print('Class weights', class_weights)

    print('\n=== Step 4: Train model ===')
    model, history_head, history_ft = train_module.train_model(
        model, base_model, train_ds, val_ds, class_weights, initial_epochs=1, fine_tune_epochs=1
    )

    print('\n=== Step 5: Save model ===')
    saved_path = train_module.save_model(model, path=WORKDIR / 'onycoscan_weighted.keras')
    print('Saved model to', saved_path)

    print('\n=== Step 6: Evaluate model ===')
    y_pred, y_true, report = train_module.evaluate_model(model, val_ds, class_names=class_names)
    print(report)

    print('\n=== Step 7: RAG sample guidance ===')
    advice = rag_module.retrieve_guidance('Acral_Melanoma Hutchinson sign')
    print(advice)

    print('\nDone. Interface can be started at Model 2/interface.py in notebook mode.')


if __name__ == '__main__':
    run_training_pipeline()
