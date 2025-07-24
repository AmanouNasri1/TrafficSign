import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        """
        Dataset for loading GTSRB traffic sign images.

        Args:
            root_dir (str): Root directory containing the dataset.
            csv_file (str, optional): Path to the CSV file with image paths and labels.
            transform (callable, optional): Transformations to apply to each image.
        """
        self.root_dir = root_dir
        self.transform = transform

        if csv_file:
            data = pd.read_csv(csv_file, sep=';')
            self.img_paths = [os.path.join(root_dir, path) for path in data['Filename']]
            self.labels = data['ClassId'].tolist()
        else:
            self.img_paths = []
            self.labels = []
            for class_id in sorted(os.listdir(root_dir)):
                class_dir = os.path.join(root_dir, class_id)
                for img_file in sorted(os.listdir(class_dir)):
                    self.img_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(int(class_id))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
