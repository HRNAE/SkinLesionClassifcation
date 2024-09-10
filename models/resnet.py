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
    '/root/stanfordData4321/stanfordData4321/standardized_images/images1',
    '/root/stanfordData4321/stanfordData4321/standardized_images/images2',
    '/root/stanfordData4321/stanfordData4321/standardized_images/images3',
    '/root/stanfordData4321/stanfordData4321/standardized_images/images4'
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
augmented_dataset = AugmentedImageDataset(ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform), './augmented_images2', transform)
print(f"Total images in augmented dataset: {len(augmented_dataset)}")

# Train and test split
train_size = int(0.8 * len(augmented_dataset))
test_size = len(augmented_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(augmented_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load pre-trained DenseNet model and modify final layer
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)

num_ftrs = net.classifier.in_features
net.classifier = nn.Linear(num_ftrs, len(categories))

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Optuna optimization
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.5, 0.9)
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1):  # Fewer epochs for optimization
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
study.optimize(objective, n_trials=1)

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
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test dataset: {100 * correct / total:.2f}%')

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((700, 700)),  # Resize to 700x700
    transforms.ToTensor(),
])
import numpy as np
import torch

def generate_occlusion_sensitivity_map(image, model, occlusion_size=15, occlusion_stride=15):
    """
    Generate an occlusion sensitivity map for the given image and model.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W).
        model (torch.nn.Module): Trained model.
        occlusion_size (int): Size of the occlusion window.
        occlusion_stride (int): Stride of the occlusion window.

    Returns:
        np.ndarray: Sensitivity map of the same size as the input image.
    """
    # Set model to evaluation mode
    model.eval()

    # Get original image size
    if len(image.size()) == 5:  # If it's a 5D tensor
        image = image.squeeze(1)  # Remove the extra dimension
    _, _, h, w = image.size()

    # Get the prediction for the original image
    with torch.no_grad():
        original_output = model(image)
    original_class = original_output.argmax(dim=1).item()

    # Initialize sensitivity map
    sensitivity_map = np.zeros((h, w))

    # Occlude part of the image and get model output for each occlusion
    for i in range(0, h, occlusion_stride):
        for j in range(0, w, occlusion_stride):
            # Create a copy of the original image
            occluded_image = image.clone()

            # Apply occlusion (e.g., zero out a region)
            occluded_image[:, :, i:i + occlusion_size, j:j + occlusion_size] = 0

            # Get model prediction for the occluded image
            with torch.no_grad():
                occluded_output = model(occluded_image)
            occluded_score = occluded_output[0, original_class].item()

            # Fill the sensitivity map with the difference in score
            sensitivity_map[i:i + occlusion_size, j:j + occlusion_size] = original_output[0, original_class].item() - occluded_score

    # Normalize the sensitivity map
    sensitivity_map = (sensitivity_map - np.min(sensitivity_map)) / (np.max(sensitivity_map) - np.min(sensitivity_map))
    sensitivity_map = (sensitivity_map * 255).astype(np.uint8)

    return sensitivity_map


class ClusterImageDataset(Dataset):
    def __init__(self, cluster_dir, transform=None):
        """
        Initialize the dataset with the directory containing images.
        
        Args:
            cluster_dir (str): Path to the cluster folder.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.cluster_dir = cluster_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        
    def _get_image_paths(self):
        """
        Get all image paths in the cluster directory.
        
        Returns:
            list: A list of tuples where each tuple contains the image path and the folder name (cluster).
        """
        image_paths = []
        for root, _, files in os.walk(self.cluster_dir):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):  # Add other extensions if needed
                    img_path = os.path.join(root, file)
                    image_paths.append(img_path)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its path.
        
        Args:
            idx (int): Index of the image.
        
        Returns:
            tuple: (image, img_path) where image is the transformed image and img_path is its file path.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_path


# Path to the Clusters directory
clusters_path = '/root/stanfordData4321/stanfordData4321/clusters'
cluster_folders = [os.path.join(clusters_path, d) for d in os.listdir(clusters_path) if os.path.isdir(os.path.join(clusters_path, d))]

# Initialize dictionaries to track accuracy for each cluster
correct_predictions = {cluster: 0 for cluster in cluster_folders}
total_images = {cluster: 0 for cluster in cluster_folders}

# Create directory for saving sensitivity maps if it doesn't exist
if not os.path.exists('./sensitivity_mapsRes'):
    os.makedirs('./sensitivity_mapsDense')

# Loop through clusters
for cluster in cluster_folders:
    print(f"Processing cluster: {os.path.basename(cluster)}")
    
    # Create dataset and loader for the current cluster
    cluster_dataset = ClusterImageDataset(cluster, transform=transform)
    cluster_loader = DataLoader(cluster_dataset, batch_size=1, shuffle=False)
    
    correct = 0
    total = 0
    image_count = 0
    
    for images, img_paths in cluster_loader:
        images = images.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
        
        # Check if prediction is correct
        cluster_number = os.path.basename(cluster).split('_')[-1]  # Extract the number from the folder name
        if predicted.item() == int(cluster_number):
            correct += 1
        total += 1

        # Generate sensitivity map for first two images in the cluster
        if image_count < 2:
            # Add unsqueeze(0) to make the image batch size 1
            sensitivity_map = generate_occlusion_sensitivity_map(images.unsqueeze(0), net)
            sensitivity_map = cv2.resize(sensitivity_map, (700, 700))

            # Read and resize overlay image
            overlay_image = cv2.imread(img_paths[0])
            if overlay_image is None:
                print(f"Warning: Could not read image {img_paths[0]}")
                continue

            overlay_image = cv2.resize(overlay_image, (700, 700))
            heatmap = cv2.applyColorMap(sensitivity_map, cv2.COLORMAP_JET)
            result = cv2.addWeighted(overlay_image, 0.7, heatmap, 0.3, 0)
            
            # Save the result
            result_path = f"./sensitivity_maps/{os.path.basename(cluster)}_image_{image_count}.png"
            cv2.imwrite(result_path, result)
            print(f"Sensitivity map saved for image {image_count} in cluster {os.path.basename(cluster)}")
        
        image_count += 1

    # Update accuracy for current cluster
    correct_predictions[cluster] += correct
    total_images[cluster] += total
    
    print(f"Cluster {os.path.basename(cluster)} accuracy: {100 * correct / total:.2f}%")


# Calculate and print overall accuracy across clusters
overall_correct = sum(correct_predictions.values())
overall_total = sum(total_images.values())
overall_accuracy = 100 * overall_correct / overall_total
print(f"Overall accuracy across all clusters: {overall_accuracy:.2f}%")
