import numpy as np

train_images = np.load('data/train_images.npy')
print("Shape of train_images:", train_images.shape)

train_labels = np.load('data/train_labels.npy')
print("Shape of train_labels:", train_labels.shape)

val_images = np.load('data/val_images.npy')
print("Shape of val_images:", val_images.shape)

val_labels = np.load('data/val_labels.npy')
print("Shape of val_labels:", val_labels.shape)