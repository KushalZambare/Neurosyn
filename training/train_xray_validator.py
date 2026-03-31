import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'xray_validation')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'xray_classifier.h5')

IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 10

def train_validator():
    
    if not os.path.exists(DATA_DIR):
        print(f"Dataset directory NOT FOUND: {DATA_DIR}")
        return

    # 1. Setup Data Generators 
    # X-rays are usually normalized to [0,1] or [-1,1]
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # 20% for validation
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 2. Load Images from Directory 
    print("Loading data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        classes=['non_xray', 'xray'], 
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        classes=['non_xray', 'xray'],
        shuffle=False
    )

    # 3. Model Architecture
    model = models.Sequential([
        
        layers.Input(shape=(224, 224, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid') 
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 4. Handle Sample Imbalance
    class_0_weight = 3616 / 640
    class_weights = {0: class_0_weight, 1: 1.0}
    
    print(f"Applying Class Weights: 0 (non_xray): {class_0_weight:.2f}, 1 (xray): 1.00")

    # 5. Execute Training
    print("Starting training on X-ray validator model...")
    history = model.fit(
        train_generator,    
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1
    )
  
    print(f"\nTraining Complete! Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    model.save(MODEL_SAVE_PATH)
    print(f"Saved binary classifier to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_validator()
