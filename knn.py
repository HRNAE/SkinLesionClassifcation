import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import models
from PIL import Image
import pandas as pd
import logging
#import optuna
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

def save_clusters(dataloader, labels, output_cluster_dir):
    print("Saving clusters...")
    if not os.path.exists(output_cluster_dir):
        os.makedirs(output_cluster_dir)

    batch_idx = 0
    for images, _ in dataloader:
        for i, img in enumerate(images):
            cluster_label = labels[batch_idx * len(images) + i]
            cluster_dir = os.path.join(output_cluster_dir, f"cluster_{cluster_label}")
            if not os.path.exists(cluster_dir):
                os.makedirs(cluster_dir)
            
            img_path = os.path.join(cluster_dir, f"img_{batch_idx}_{i}.png")
            save_image(img, img_path)
        batch_idx += 1

    print(f"Clusters saved to {output_cluster_dir}")

if __name__ == "__main__":
    # Step 1: Load datasets
    root_dirs = ["/root/stanfordData4321/standardized_images/images4", 
                 "/root/stanfordData4321/standardized_images/images3", 
                 "/root/stanfordData4321/standardized_images/images2", 
                 "/root/stanfordData4321/standardized_images/images1"]
    

    print("Loading original dataset...")
    original_dataset = ExcelImageDataset('./dataRef/release_midas.xlsx', root_dirs, transform)
    print("Original dataset loaded.")

    #print("Loading augmented dataset...")
    #augmented_dataset = AugmentedImageDataset(original_dataset, augmented_dirs, transform=transform)
    dataloader = DataLoader(original_dataset, batch_size=32, shuffle=False)
    #print("Augmented dataset loaded.")

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
    n_clusters = 14
    print(f"Performing KNN clustering with {n_clusters} clusters...")
    clustered_images, centroids, labels = knn_clustering(reduced_features, n_clusters)

    # Step 5: Save the clusters
    output_cluster_dir = "./clustersNew"
    save_clusters(dataloader, labels, output_cluster_dir)
