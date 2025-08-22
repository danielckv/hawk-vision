# train_model.py

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

from humanskinlib import get_current_gpu

print(f"PyTorch Version: {torch.__version__}")

# --- Configuration ---
DATASET_PATH = 'hmnist_64_64_RGB.csv'
MODEL_SAVE_PATH = 'best_densenet_skin_lesion_model.pth'
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 64
NUM_CLASSES = 7  # Based on the HAM10000 dataset


# --- 1. Custom Dataset Class ---
class SkinLesionDataset(Dataset):
    """Custom Dataset for loading the skin lesion data from a CSV file."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with pixel values and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the dataset
        dataframe = pd.read_csv(csv_file)

        # Separate features (pixels) and labels
        self.features = dataframe.drop('label', axis=1).values
        self.labels = dataframe['label'].values

        # Encode string labels to integers
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)

        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Reshape the flat pixel array into a 3D image (H x W x C)
        # 12288 = 64 * 64 * 3
        image = self.features[idx].reshape(IMAGE_SIZE, IMAGE_SIZE, 3).astype('uint8')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


# --- 2. Data Transforms and Augmentation ---
# Define transforms for training (with augmentation) and validation (only normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# --- 3. Main Training Function ---
def train():
    """Main function to orchestrate the model training and validation process."""

    # Check for dataset existence
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at '{DATASET_PATH}'")
        print("Please download 'hmnist_64_64_RGB.csv' and place it in the same directory.")
        return

    # Set device
    device = get_current_gpu()

    # --- Load and Split Data ---
    full_dataset = SkinLesionDataset(csv_file=DATASET_PATH)

    # Split the dataset into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the respective transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # --- Initialize Model, Loss, and Optimizer ---
    # Load a pre-trained DenseNet121 and modify the final layer
    model = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)  # Adjust for our number of classes

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    since = time.time()
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'Best validation accuracy: {best_acc:.4f}. Model saved to {MODEL_SAVE_PATH}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # --- Plotting Training History ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('training_history.png')
    plt.show()


# --- Main Execution Guard ---
if __name__ == '__main__':
    train()