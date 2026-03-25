from system.imports.imports import MobileNetV2, GlobalAveragePooling2D, Dense, Dropout, Model, Adam

# ==========================================
# 3. MODEL ARCHITECTURE (Transfer Learning)
# ==========================================
def build_model(num_classes):
    """
    Builds a Deep Learning model using MobileNetV2.
    Transfer Learning is used to achieve high accuracy (>95%) on small medical datasets.
    """
    print("\n🏗️ Building Deep Learning Model (MobileNetV2)...")

    # Load pre-trained MobileNetV2 (trained on ImageNet)
    # We exclude the top layers to add our own specific to Nail Diseases
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model initially to keep pre-trained features
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x) # Dropout reduces overfitting
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
