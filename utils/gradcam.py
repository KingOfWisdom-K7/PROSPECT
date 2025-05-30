import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from config import config

def generate_heatmap(model, img_array, layer_name="top_conv"):
    """
    Generates Grad-CAM heatmap for model interpretability
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, H, W, 3)
        layer_name: Target convolutional layer name
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()
    
    heatmap = np.maximum(heatmap, 0)
    heatmap = cv2.resize(heatmap, config.IMG_SIZE)
    return heatmap / np.max(heatmap)

def visualize_explanation(model, image_path):
    """Complete Grad-CAM visualization pipeline"""
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, config.IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_preprocessed = tf.expand_dims(img_rgb, axis=0) / 255.0
    
    # Generate heatmap
    heatmap = generate_heatmap(model, img_preprocessed)
    
    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
    
    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title("Grad-CAM Explanation")
    plt.axis('off')
    plt.show()