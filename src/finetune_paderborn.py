# src/finetune_paderborn.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from preprocess_paderborn import preprocess_paderborn

# --------------------------------------------------
# Load and preprocess Paderborn data
# --------------------------------------------------
PAD_ROOT = "../data/paderborn"

X, y = preprocess_paderborn(PAD_ROOT)
y = to_categorical(y, num_classes=4)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y.argmax(axis=1),
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp.argmax(axis=1),
    random_state=42
)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# --------------------------------------------------
# Load pretrained CWRU model
# --------------------------------------------------
model = load_model("../models/cwru_model.keras")

# --------------------------------------------------
# Freeze early layers (CRITICAL)
# --------------------------------------------------
# --------------------------------------------------
# Freeze / unfreeze layers for fine-tuning
# --------------------------------------------------

# First freeze everything
for layer in model.layers:
    layer.trainable = False
    
# Unfreeze last THREE convolution blocks + classifier
for layer in model.layers[-20:]:
    layer.trainable = True



print("\nTrainable layers:")
for layer in model.layers:
    print(layer.name, layer.trainable)

# --------------------------------------------------
# Compile with LOW learning rate
# --------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# Fine-tuning
# --------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Fine-tuned Test Accuracy:", test_acc)

# --------------------------------------------------
# Save fine-tuned model
# --------------------------------------------------
model.save("../models/cwru_paderborn_ft.keras")
print("Fine-tuned model saved as cwru_paderborn_ft.keras")
