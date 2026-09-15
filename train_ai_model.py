import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ==============================
# PCB VISION - AI MODEL TRAINING
# ==============================

TRAIN_DIR = "ai_defect_dataset/train"
VALIDATION_DIR = "ai_defect_dataset/validation"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

CLASS_NAMES = [
    "Copper",
    "Mousebite",
    "Open",
    "Pin-hole",
    "Short",
    "Spur"
]

print("=" * 60)
print("PCB VISION - AI MODEL TRAINING")
print("=" * 60)

# Check dataset
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"Validation folder not found: {VALIDATION_DIR}"
    )

# ==============================
# LOAD TRAINING DATA
# ==============================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# ==============================
# LOAD VALIDATION DATA
# ==============================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:")
print(train_dataset.class_names)

# ==============================
# PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ==============================
# CNN MODEL
# ==============================

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(6, activation="softmax")
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# ==============================
# CALLBACKS
# ==============================

checkpoint = ModelCheckpoint(
    "pcb_defect_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    mode="max",
    verbose=1
)

# ==============================
# TRAINING
# ==============================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

# ==============================
# SAVE FINAL MODEL
# ==============================

model.save("pcb_defect_model.keras")

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved:")
print("pcb_defect_model.keras")

print(
    f"\nFinal Training Accuracy: "
    f"{history.history['accuracy'][-1] * 100:.2f}%"
)

print(
    f"Final Validation Accuracy: "
    f"{history.history['val_accuracy'][-1] * 100:.2f}%"
)

print("\nPCB VISION AI model is ready!")
