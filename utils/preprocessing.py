import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE
import config

def fix_shape(images, labels):
    if images.shape[0] != labels.shape[0]:
        images = np.transpose(images, (2, 0, 1))
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return images

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    if tf.random.uniform(()) > 0.5:
        zoom_factor = tf.random.uniform(shape=[], minval=1.0, maxval=1.1)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
        zoomed = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(zoomed, h, w)
    return image, label

def create_dataset(images, labels, shuffle=False, apply_smote=False):
    images = fix_shape(images, labels)
    resized = tf.image.resize(images, config.IMG_SIZE)
    if resized.shape[-1] == 1:
        resized = tf.tile(resized, [1, 1, 1, 3])

    if apply_smote:
        reshaped = resized.numpy().reshape(resized.shape[0], -1)
        sm = SMOTE(random_state=config.RANDOM_SEED)
        X_resampled, y_resampled = sm.fit_resample(reshaped, labels)
        resized = X_resampled.reshape(-1, *config.IMG_SIZE, 3)
        labels = y_resampled

    dataset = tf.data.Dataset.from_tensor_slices((resized, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    return dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_augmented_dataset(images, labels, shuffle=False, apply_smote=False):
    images = fix_shape(images, labels)
    resized = tf.image.resize(images, config.IMG_SIZE)
    if resized.shape[-1] == 1:
        resized = tf.tile(resized, [1, 1, 1, 3])
    dataset = tf.data.Dataset.from_tensor_slices((resized, labels))
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    return dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def prepare_datasets(use_smote=False):
    train_images = np.load(f'{config.DATA_DIR}/train_images.npy')
    train_labels = np.load(f'{config.DATA_DIR}/train_labels.npy')
    val_images = np.load(f'{config.DATA_DIR}/val_images.npy')
    val_labels = np.load(f'{config.DATA_DIR}/val_labels.npy')
    train_ds = create_augmented_dataset(train_images, train_labels.flatten(), shuffle=True, apply_smote=use_smote)
    val_ds = create_dataset(val_images, val_labels.flatten())
    return train_ds, val_ds
