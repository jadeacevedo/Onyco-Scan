from system.imports.imports import EarlyStopping, ReduceLROnPlateau, Adam, np, plt, sns, classification_report, confusion_matrix

# ==========================================
# 4. TRAINING & EVALUATION
# ==========================================
def train_and_evaluate(model, train_gen, val_gen):
    """Trains the model with Early Stopping to prevent overfitting."""

    # Callbacks for optimization
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]

    print("\n🚀 Starting Training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # --- Final Evaluation ---
    print("\n📊 Evaluating on Validation Set...")
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"🏆 Final Validation Accuracy: {val_acc * 100:.2f}%")

    # --- Fine Tuning (Optional but recommended for 95%+) ---
    if val_acc < 0.95:
        print("\n⚠️ Accuracy under 95%. Initiating Fine-Tuning Phase...")
        base_model = model.layers[0]
        base_model.trainable = True

        # Fine-tune only the top 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        history_fine = model.fit(
            train_gen,
            epochs=10,
            validation_data=val_gen,
            callbacks=callbacks
        )
        val_loss, val_acc = model.evaluate(val_gen)
        print(f"🏆 Final Accuracy after Fine-Tuning: {val_acc * 100:.2f}%")

    return model, history, val_acc

def generate_report(model, val_gen):
    """Generates a classification report and confusion matrix."""
    print("\n📑 Generating Classification Report...")

    # Get predictions
    val_gen.reset()
    Y_pred = model.predict(val_gen)
    y_pred = np.argmax(Y_pred, axis=1)

    # Get true labels
    class_labels = list(val_gen.class_indices.keys())

    # Print Report
    print(classification_report(val_gen.classes, y_pred, target_names=class_labels))

    # Plot Confusion Matrix
    cm = confusion_matrix(val_gen.classes, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Nail Disease Confusion Matrix')
    plt.show()
