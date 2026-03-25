import numpy as np
import pathlib
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers, models, applications


def compute_class_weights(dataset):
    labels = []
    for _, label in dataset.unbatch():
        labels.append(int(label.numpy()))

    labels = np.array(labels)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))
    return class_weights


def build_model(img_size=224, num_classes=3, dropout=0.4):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = applications.densenet.preprocess_input(x)

    base_model = applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model, base_model


def train_model(model, base_model, train_ds, val_ds, class_weights, initial_epochs=12, fine_tune_epochs=15):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return model, history_head, history_ft


def save_model(model, path='onycoscan_weighted.keras'):
    model.save(str(path))
    saved = pathlib.Path(path)
    if saved.suffix == '.keras':
        h5_path = saved.with_suffix('.h5')
        model.save(str(h5_path))
    elif saved.suffix == '.h5':
        keras_path = saved.with_suffix('.keras')
        model.save(str(keras_path))
    return str(path)


def evaluate_model(model, val_ds, class_names=None):
    y_pred = []
    y_true = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    report = None
    if class_names is not None:
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_true, y_pred, target_names=class_names)

    return y_pred, y_true, report
