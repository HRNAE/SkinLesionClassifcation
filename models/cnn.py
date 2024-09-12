import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.utils import save_image
from PIL import Image
import pandas as pd
import logging
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Using device: %s", device)

# Load the Excel file
excel_file_path = './dataRef/release_midas.xlsx'
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"{excel_file_path} does not exist. Please check the path.")

df = pd.read_excel(excel_file_path)
logging.info("Excel file loaded. First few rows:")
logging.info("%s", df.head())

# Define categories and image size
categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']
img_size = 224  # Image size set to 224

# Compose the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset class to load images from Excel and directories
class ExcelImageDataset(Dataset):
    def __init__(self, excel_file, root_dirs, transform=None):
        self.data_frame = pd.read_excel(excel_file)
        self.data_frame.iloc[:, 0] = self.data_frame.iloc[:, 0].astype(str)
        self.root_dirs = root_dirs
        self.transform = transform

        self.label_map = {label: idx for idx, label in enumerate(categories)}
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        valid_paths = []
        for idx, row in self.data_frame.iterrows():
            img_found = False
            for root_dir in self.root_dirs:
                img_name = os.path.join(root_dir, row['midas_file_name'])
                if os.path.isfile(img_name):
                    label = row['clinical_impression_1']
                    if label not in self.label_map:
                        logging.warning("Label '%s' not in label_map.", label)
                        continue
                    valid_paths.append((img_name, label))
                    img_found = True
                    break
            if not img_found:
                logging.warning("Image %s not found in any root directory.", row['midas_file_name'])
        logging.info("Total valid paths found: %d", len(valid_paths))
        return valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name, label = self.image_paths[idx]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label_map.get(label, -1), dtype=torch.long)
        return image, label

# Dataset class to load augmented images from the directory
class AugmentedImageDataset(Dataset):
    def __init__(self, original_dataset, augmented_dir, transform=None):
        self.original_dataset = original_dataset
        self.augmented_dir = augmented_dir
        self.transform = transform
        self.augmented_paths = self._get_augmented_paths()

    def _get_augmented_paths(self):
        augmented_paths = []
        for root, _, files in os.walk(self.augmented_dir):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    label = int(os.path.basename(root))
                    augmented_paths.append((img_path, label))
        return augmented_paths

    def __len__(self):
        return len(self.augmented_paths)

    def __getitem__(self, idx):
        img_path, label = self.augmented_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Define the root directories
root_dirs = [
    os.path.join(os.getcwd(), '/root/stanfordData4321/stanfordData4321/standardized_images/images1'),
    os.path.join('/root/stanfordData4321/stanfordData4321/standardized_images/images2'),
    os.path.join('/root/stanfordData4321/stanfordData4321/standardized_images/images3'),
    os.path.join('/root/stanfordData4321/stanfordData4321/standardized_images/images4')
]

# Load original dataset
logging.info("Creating combined dataset...")
dataset = ExcelImageDataset(excel_file_path, root_dirs, transform)

# Load augmented dataset
augmented_dataset = AugmentedImageDataset(dataset, './augmented_images', transform)

# Combined dataset
class CombinedDataset(Dataset):
    def __init__(self, original_dataset, augmented_dataset):
        self.original_dataset = original_dataset
        self.augmented_dataset = augmented_dataset

    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_dataset)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            return self.augmented_dataset[idx - len(self.original_dataset)]

# Create the combined dataset
combined_dataset = CombinedDataset(dataset, augmented_dataset)
logging.info(f"Total images in combined dataset: {len(combined_dataset)}")

# Split dataset into training and testing subsets
logging.info("Splitting dataset into training and testing subsets...")
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = random_split(augmented_dataset, [train_size, test_size])

logging.info("Length train dataset: %d", len(train_dataset))
logging.info("Length test dataset: %d", len(test_dataset))

# Define the custom model class
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(categories))

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model instance, loss function, and optimizer
model = CustomModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set up data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# Train the model
logging.info("Starting model training...")
train_model(model, train_loader, criterion, optimizer, epochs=25)

# Evaluate the model
logging.info("Evaluating model...")
true_labels, pred_labels = evaluate_model(model, test_loader)

# Calculate metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')

# Log metrics
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")

# Function to plot sample predictions
def plot_sample_predictions(data_loader, model, num_samples=5):
    model.eval()
    images, labels = next(iter(data_loader))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    fig = plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i+1, xticks=[], yticks=[])
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        true_label = labels[i].item()
        pred_label = preds[i].item()
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
    plt.show()

# Plot sample predictions
plot_sample_predictions(test_loader, model)

# Save the model
model_path = 'best_possible_model.pth'
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")
