# inference.py

import os

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

from humanskinlib import get_current_gpu

print(f"PyTorch Version: {torch.__version__}")

# --- Configuration ---
MODEL_PATH = 'best_densenet_skin_lesion_model.pth'
DATASET_PATH = 'hmnist_64_64_RGB.csv'  # Needed to grab a sample image
IMAGE_SIZE = 64
NUM_CLASSES = 7

# --- Class Labels Mapping ---
# This mapping is based on the HAM10000 dataset, which hmnist_64_64_RGB.csv is derived from.
# The labels are sorted alphabetically to match the LabelEncoder from training.
CLASS_NAMES = [
    'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease (akiec)',
    'Basal cell carcinoma (bcc)',
    'Benign keratosis-like lesions (bkl)',
    'Dermatofibroma (df)',
    'Melanocytic nevi (nv)',
    'Melanoma (mel)',
    'Vascular lesions (vasc)'
]
# The original labels in the CSV are:
# akiec, bcc, bkl, df, mel, nv, vasc
# LabelEncoder sorts them alphabetically, so the mapping is:
# 0: akiec, 1: bcc, 2: bkl, 3: df, 4: mel, 5: nv, 6: vasc
# Let's adjust our class names to match this order.
CLASS_NAMES_MAPPING = {
    0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'
}

# --- 1. Load Model ---
def load_trained_model(model_path):
    """Loads the pre-trained DenseNet model with the saved state dictionary."""

    # Initialize the model architecture
    model = models.densenet121(weights=None)  # We don't need pre-trained weights, we have our own
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)

    # Load the saved state dictionary

    model.load_state_dict(torch.load(model_path, map_location=get_current_gpu()))

    # Set the model to evaluation mode
    model.eval()
    return model


# --- 2. Define Image Transformation ---
# Use the same transformations as the validation set during training
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 3. Prediction Function ---
def predict(model, image_pixels, device):
    """
    Takes a model and a flat array of image pixels, preprocesses it,
    and returns the predicted class and confidence.
    """
    # Reshape the flat pixel array into a 3D image (H x W x C)
    image = image_pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype('uint8')

    # Apply transformations
    image_tensor = inference_transform(image).unsqueeze(0)  # Add batch dimension

    # Move tensor to the correct device
    image_tensor = image_tensor.to(device)
    model = model.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)

        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get the top class
        top_prob, top_catid = torch.topk(probabilities, 1)

        predicted_idx = top_catid.item()
        confidence = top_prob.item()

    return CLASS_NAMES_MAPPING[predicted_idx], confidence


# --- Main Execution Block ---
def main():
    """Main function to run the inference example."""
    print("--- Running Inference ---")

    # Check for required files
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please run 'train_model.py' first to train and save the model.")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at '{DATASET_PATH}'")
        print("Inference script needs this file to get a sample image.")
        return

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the trained model
    model = load_trained_model(MODEL_PATH)
    print("Model loaded successfully.")

    # 2. Get a sample image from the dataset for demonstration
    dataframe = pd.read_csv(DATASET_PATH)

    # Select a random sample
    sample = dataframe.sample(1)
    image_pixels = sample.drop('label', axis=1).values[0]
    true_label = sample['label'].values[0]

    print(f"\nSelected a random sample for prediction.")
    print(f"--> True Label: '{true_label}'")

    # 3. Make a prediction
    predicted_label, confidence = predict(model, image_pixels, device)

    print(f"--> Predicted Label: '{predicted_label}'")
    print(f"--> Confidence: {confidence:.2%}")

    if predicted_label == true_label:
        print("\n✅ Prediction is correct!")
    else:
        print("\n❌ Prediction is incorrect.")


if __name__ == '__main__':
    main()