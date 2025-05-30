import tensorflow as tf
import numpy as np
from utils.preprocessing import create_dataset
import config
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
from model import f1_score

def predict(model_name='best_model_smote.keras'): # Default to SMOTE trained model
    print("--- Starting Prediction and Evaluation Script ---")

    print(f"Loading the best model: {config.MODEL_DIR}/{model_name}...")
    try:
        model = tf.keras.models.load_model(
            os.path.join(config.MODEL_DIR, model_name),
            custom_objects={'f1_score': f1_score}
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Expected model path: {os.path.join(config.MODEL_DIR, model_name)}")
        return

    print("Loading test data...")
    try:
        test_images = np.load(f'{config.DATA_DIR}/test_images.npy')
        test_labels = np.load(f'{config.DATA_DIR}/test_labels.npy')
        print(f"Raw test_images shape: {test_images.shape}")
        print(f"Raw test_labels shape: {test_labels.shape}")
    except FileNotFoundError:
        print("Error: test_images.npy or test_labels.npy not found.")
        return
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    print("Creating test dataset...")
    try:
        test_ds = create_dataset(test_images, test_labels.flatten(), shuffle=False)
        print("Test dataset created.")
    except Exception as e:
        print(f"Error creating test dataset: {e}")
        return

    print("Making predictions on the entire test set...")
    try:
        predictions = model.predict(test_ds)
        predicted_classes = (predictions > 0.5).astype("int32").flatten()
        true_labels = test_labels.flatten()
        print("Predictions generated.")

        print("\n--- First 100 Predictions ---")
        num_samples_to_show = min(100, len(predicted_classes))
        for i in range(num_samples_to_show):
            print(f"Predicted: {predicted_classes[i]}, True: {true_labels[i]}")

        print("\n--- Classification Report ---")
        print(classification_report(true_labels, predicted_classes))

        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(true_labels, predicted_classes)
        print(cm)

        # --- ROC Curve ---
        fpr, tpr, thresholds_roc = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'roc_curve.png'))
        print(f"\nROC curve saved to {config.OUTPUT_DIR}/roc_curve.png")
        plt.close()

        # --- Precision-Recall Curve ---
        precision, recall, thresholds_pr = precision_recall_curve(true_labels, predictions)
        avg_precision = average_precision_score(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(config.OUTPUT_DIR, 'pr_curve.png'))
        print(f"Precision-Recall curve saved to {config.OUTPUT_DIR}/pr_curve.png")
        plt.close()

    except Exception as e:
        print(f"Error during prediction or evaluation: {e}")

if __name__ == '__main__':
    predict(model_name='best_model_augmented.keras')