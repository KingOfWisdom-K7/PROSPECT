import numpy as np
import tensorflow as tf
from utils.preprocessing import prepare_datasets
from model import build_model
import config
from sklearn.utils.class_weight import compute_class_weight

def train_model(use_smote=False):
    print("Preparing data...")
    train_ds, val_ds = prepare_datasets(use_smote=use_smote)

    if not use_smote:
        labels_np = np.concatenate([y.numpy() for _, y in train_ds])
        weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
        class_weight = dict(zip(np.unique(labels_np), weights))
    else:
        class_weight = None

    model = build_model()
    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS * 2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=config.EARLY_STOP_PATIENCE, mode='max', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                f"{config.MODEL_DIR}/best_model_smote.keras" if use_smote else f"{config.MODEL_DIR}/best_model_augmented.keras",
                monitor='val_auc', save_best_only=True, mode='max'),
            tf.keras.callbacks.CSVLogger(
                f"{config.MODEL_DIR}/training_log_smote.csv" if use_smote else f"{config.MODEL_DIR}/training_log_augmented.csv")
        ],
        class_weight=class_weight
    )
    return history

if __name__ == "__main__":
    print("--- Training with Data Augmentation ---")
    train_model(use_smote=False)
    print("--- Training with SMOTE + Augmentation ---")
    train_model(use_smote=True)
