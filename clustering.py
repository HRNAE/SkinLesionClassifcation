import os
import shutil
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ExcelImageDataset(Dataset):
    def __init__(self, excel_file, root_dirs, transform=None):
        self.data_frame = pd.read_excel(excel_file)
        self.data_frame.iloc[:, 0] = self.data_frame.iloc[:, 0].astype(str)
        self.root_dirs = root_dirs
        self.transform = transform
        self.categories = list(set(self.data_frame['clinical_impression_1'].dropna()))
        self.label_map = {label: idx for idx, label in enumerate(self.categories)}
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

# Paths and transformation
excel_file = './dataRef/release_midas.xlsx'  # Path to your Excel file
root_dirs = ['/root/stanfordData4321/stanfordData4321/standardized_images/images1',
            '/root/stanfordData4321/stanfordData4321/standardized_images/images2',
            '/root/stanfordData4321/stanfordData4321/standardized_images/images3',
            '/root/stanfordData4321/stanfordData4321/standardized_images/images4']  # Root directories containing images
output_folder = '/root/stanfordData4321/clustersNew'  # Path to save the clustered images

# Image transformations (if needed)
transform = transforms.Compose([
    transforms.Resize((700, 700)),
    transforms.ToTensor(),
])

# Initialize dataset
dataset = ExcelImageDataset(excel_file, root_dirs, transform=transform)

# Create directories for each label/cluster inside your GitHub repository
for label_name in dataset.label_map.keys():
    label_dir = os.path.join(output_folder, label_name)
    os.makedirs(label_dir, exist_ok=True)
    print(f"Directory created: {label_dir}")

# Copy images to their respective cluster folders based on label
for img_name, label in dataset.image_paths:
    cluster_dir = os.path.join(output_folder, label)  # Label is directly used as folder name
    try:
        # Ensure the image is copied
        shutil.copy(img_name, os.path.join(cluster_dir, os.path.basename(img_name)))
        print(f"Image {img_name} copied to {cluster_dir}")
    except Exception as e:
        print(f"Error copying {img_name} to {cluster_dir}: {e}")

print(f"Images have been clustered and saved to {output_folder}.")
