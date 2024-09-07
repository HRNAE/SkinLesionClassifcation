import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import pandas as pd
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info("Using device: %s", device)

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

class CustomImageDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.root_dirs = root_dirs
        self.transform = transform
        print("Loading image paths...")
        self.image_paths = self._get_image_paths()
        print("Image paths loaded. Total images: %d" % len(self.image_paths))

    def _get_image_paths(self):
        valid_paths = []
        for root_dir in self.root_dirs:
            for label_dir in os.listdir(root_dir):
                label_path = os.path.join(root_dir, label_dir)
                if os.path.isdir(label_path):
                    label = int(label_dir)  # Assuming folder names are integers corresponding to labels
                    for file in os.listdir(label_path):
                        if file.endswith(".png"):
                            img_path = os.path.join(label_path, file)
                            valid_paths.append((img_path, label))
        logging.info("Total valid paths found: %d", len(valid_paths))
        return valid_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def extract_features(model, dataloader):
    features = []
    print("Extracting features...")
    total_images = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            output = model(images.to(device))
            features.append(output.squeeze().cpu().numpy())
            
            # Print progress every 500 images
            if (batch_idx + 1) * batch_size % 500 == 0 or (batch_idx + 1) * batch_size == total_images:
                progress = (batch_idx + 1) * batch_size
                print(f"Processed {progress} images out of {total_images}")

    print("Feature extraction complete. Total features: %d" % len(features))
    return np.vstack(features)

def reduce_dimensionality(features, n_components=50):
    print("Reducing dimensionality...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    print("Dimensionality reduction complete. Reduced features shape: %s" % str(reduced_features.shape))
    return reduced_features

def initialize_clusters(reduced_features, n_clusters):
    print("Initializing clusters...")
    random_indices = random.sample(range(len(reduced_features)), n_clusters)
    centroids = reduced_features[random_indices]
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(centroids)
    _, labels = knn.kneighbors(reduced_features)
    print("Clusters initialized.")
    return labels.flatten(), centroids

def update_centroids(clustered_images, reduced_features, n_clusters):
    print("Updating centroids...")
    new_centroids = []
    for i in range(n_clusters):
        cluster_indices = clustered_images[i]
        if len(cluster_indices) > 0:
            cluster_features = reduced_features[cluster_indices]
            new_centroids.append(cluster_features.mean(axis=0))
        else:
            new_centroids.append(np.zeros(reduced_features.shape[1]))
    print("Centroids updated.")
    return np.array(new_centroids)

def knn_clustering(reduced_features, n_clusters, max_iterations=10):
    print("Performing KNN clustering...")
    labels, centroids = initialize_clusters(reduced_features, n_clusters)
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        clustered_images = {i: [] for i in range(n_clusters)}
        for i, label in enumerate(labels):
            clustered_images[label].append(i)

        new_centroids = update_centroids(clustered_images, reduced_features, n_clusters)
        if np.allclose(centroids, new_centroids):
            print("Convergence reached.")
            break
        centroids = new_centroids

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(centroids)
        _, labels = knn.kneighbors(reduced_features)
        labels = labels.flatten()

    print("KNN clustering complete.")
    return clustered_images, centroids, labels

def save_clusters_to_csv(dataloader, labels, output_csv):
    print("Saving clusters to CSV...")
    rows = []

    for idx, (img_path, _) in enumerate(dataloader.dataset.image_paths):
        cluster_label = labels[idx]
        rows.append([img_path, cluster_label])

    # Convert to a DataFrame and save as CSV
    df = pd.DataFrame(rows, columns=['file_name', 'cluster'])
    df.to_csv(output_csv, index=False)
    print(f"Clusters saved to {output_csv}")

if __name__ == "__main__":
    # Step 1: Load datasets
    root_dirs = ["images1", "images2", "images3", "images4"]

    print("Loading dataset...")
    original_dataset = CustomImageDataset(root_dirs, transform=transform)
    dataloader = DataLoader(original_dataset, batch_size=32, shuffle=False)
    print("Dataset loaded.")

    # Step 2: Feature extraction using a pre-trained ResNet18 model
    print("Loading pre-trained ResNet18 model...")
    resnet = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
    model.to(device)
    model.eval()

    print("Extracting features...")
    features = extract_features(model, dataloader)

    # Step 3: Dimensionality reduction using PCA
    reduced_features = reduce_dimensionality(features)

    # Step 4: Perform KNN clustering
    n_clusters = 10
    print(f"Performing KNN clustering with {n_clusters} clusters...")
    clustered_images, centroids, labels = knn_clustering(reduced_features, n_clusters)

    # Step 5: Save clusters to CSV
    output_csv = "./clusters.csv"
    save_clusters_to_csv(dataloader, labels, output_csv)
