import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load pretrained CNN model (ResNet18)
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the final classification layer
model.eval()

# Image transformations for the pretrained model (ResNet18 expects 224x224 images)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to your folder with 4 subfolders containing images
image_folder = "/root/stanfordData4321/stanfordData4321/standardized_images"
output_folder = "./clustersNew"  # Path to save the clustered images

# Function to load and preprocess images, and extract CNN embeddings
def load_and_extract_embeddings(folder, model):
    image_paths = []
    embeddings = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                image_paths.append(img_path)
                img = cv2.imread(img_path)
                if img is not None:
                    img_transformed = transform(img).unsqueeze(0)  # Add batch dimension
                    with torch.no_grad():
                        embedding = model(img_transformed).squeeze().numpy()  # Extract embedding
                    embeddings.append(embedding)
    return np.array(embeddings), image_paths

# Extract embeddings
embeddings, image_paths = load_and_extract_embeddings(image_folder, model)

# Optional: Reduce dimensionality with PCA
pca = PCA(n_components=50)  # Reduce to 50 dimensions
embeddings_pca = pca.fit_transform(embeddings)

# Apply K-Means for clustering
def apply_kmeans(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    return kmeans

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(2, 10):  # Try different numbers of clusters
    kmeans = apply_kmeans(embeddings_pca, k)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure()
plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the optimal number of clusters from the elbow method
optimal_k = 4  # Change this based on the elbow plot

# Final K-Means clustering
kmeans = apply_kmeans(embeddings_pca, optimal_k)
labels = kmeans.labels_

# Create directories for each cluster
for i in range(optimal_k):
    cluster_dir = os.path.join(output_folder, f'cluster_{i}')
    os.makedirs(cluster_dir, exist_ok=True)

# Save images to respective cluster folders
for img_path, label in zip(image_paths, labels):
    # Get the filename and create new path in the cluster folder
    filename = os.path.basename(img_path)
    cluster_dir = os.path.join(output_folder, f'cluster_{label}')
    shutil.copy(img_path, os.path.join(cluster_dir, filename))

print("Images have been saved to respective cluster folders.")

# Visualize the clusters (optional)
plt.figure()
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualization')
plt.colorbar(label='Cluster Label')
plt.show()
