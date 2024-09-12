import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from collections import Counter
import optuna
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.cm as cm
import torchvision.models as models


# Define categories and image size
categories = ['7-malignant-bcc', '1-benign-melanocytic nevus', '6-benign-other',
              '14-other-non-neoplastic/inflammatory/infectious', '8-malignant-scc',
              '9-malignant-sccis', '10-malignant-ak', '3-benign-fibrous papule',
              '4-benign-dermatofibroma', '2-benign-seborrheic keratosis',
              '5-benign-hemangioma', '11-malignant-melanoma',
              '13-other-melanocytic lesion with possible re-excision (severe/spitz nevus, aimp)',
              '12-malignant-other']
img_size = 224

# Define transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

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
                        print(f"Warning: Label '{label}' not in label_map.")
                        continue
                    valid_paths.append((img_name, label))
                    img_found = True
                    break
            if not img_found:
                print(f"Warning: Image {row['midas_file_name']} not found in any root directory.")
        print(f"Total valid paths found: {len(valid_paths)}")
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

# Define the root directories
root_dirs = [
    '/root/stanfordData4321/standardized_images/images1',
    '/root/stanfordData4321/standardized_images/images2',
    '/root/stanfordData4321/standardized_images/images3',
    '/root/stanfordData4321/standardized_images/images4'
]

# Augmented dataset class
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

# Create augmented dataset
augmented_dataset = AugmentedImageDataset(ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform), './augmented_images', transform)
print(f"Total images in augmented dataset: {len(augmented_dataset)}")

# Train and test split
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(augmented_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load pre-trained ResNet model and modify final layer
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(categories))

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# Optuna optimization
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):  # Fewer epochs for faster optimization
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=8)

best_params = study.best_params
print("Best parameters found by Optuna:", best_params)

# Training with the best parameters
best_lr = best_params['lr']
best_momentum = best_params['momentum']
optimizer = optim.SGD(net.parameters(), lr=best_lr, momentum=best_momentum)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):  # Adjust epoch count
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
net.eval()

all_preds = []
all_labels = []

# Iterate through the test set
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Get model predictions
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        
        # Collect predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Calculate precision, recall, f1 score, and accuracy
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
accuracy = accuracy_score(all_labels, all_preds)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")




class ClusterImageDataset(Dataset):
    def __init__(self, cluster_dir, transform=None):
        self.cluster_dir = cluster_dir
        self.image_paths = []
        self.transform = transform
        self._load_images()

    def _load_images(self):
        for root, _, files in os.walk(self.cluster_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


# Path to the Clusters directory
clusters_path = '/root/stanfordData4321/clustersNew'
cluster_folders = [os.path.join(clusters_path, d) for d in os.listdir(clusters_path) if os.path.isdir(os.path.join(clusters_path, d))]

correct_predictions = {}
total_images = {}

# Function to get label from folder name
def get_label_from_folder_name(folder_name):
    # Extracts the numeric label from folder name (e.g., '1-benign-melanocytic nevus')
    label_str = folder_name.split('-')[0]
    return int(label_str)

# Iterate over each cluster
for cluster in cluster_folders:
    cluster_name = os.path.basename(cluster)
    print(f"Processing cluster: {cluster_name}")

    # Get numeric label from the cluster folder name
    cluster_label = get_label_from_folder_name(cluster_name)
    
    # Create dataset and loader for the current cluster
    cluster_dataset = ClusterImageDataset(cluster, transform=transform)
    cluster_loader = DataLoader(cluster_dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    image_count = 0
    
    # Iterate over images in the cluster
    for images, img_paths in cluster_loader:
        images = images.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
        
        # Compare predicted label to cluster label
        if predicted.item() == cluster_label:
            correct += 1
        total += 1

        image_count += 1

    # Update accuracy for the current cluster
    correct_predictions[cluster_name] = correct
    total_images[cluster_name] = total
    
    # Print accuracy for the cluster
    if total > 0:
        print(f"Cluster {cluster_name} accuracy: {100 * correct / total:.2f}%")
    else:
        print(f"No images found for cluster {cluster_name}")

# After processing all clusters, print overall accuracy per cluster
print("\nCluster-wise accuracy:")
for cluster_name in correct_predictions:
    correct = correct_predictions[cluster_name]
    total = total_images[cluster_name]
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Cluster {cluster_name}: {accuracy:.2f}%")
