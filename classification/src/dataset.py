import os
import pandas as pd
from torch.utils.data import Dataset
import cv2

class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None, image_size=(32, 32)):
        """
        Dataset for loading GTSRB traffic sign images.

        Args:
            root_dir (str): Root directory of the dataset.
            csv_file (str, optional): Path to the CSV file (for test set).
            transform (callable, optional): Torchvision transforms.
            image_size (tuple): Desired image size (width, height).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size

        if csv_file:
            # First try reading normally
            try:
                data = pd.read_csv(csv_file, sep=';')
                # If we get a single-column DataFrame, parse manually
                if len(data.columns) == 1:
                    data = self._parse_semicolon_separated_csv(csv_file)
            except Exception as e:
                print(f"Error reading CSV: {e}")
                data = self._parse_semicolon_separated_csv(csv_file)
                
            self.img_paths = [os.path.join(root_dir, path.strip()) for path in data['Filename']]
            self.labels = data['ClassId'].astype(int).tolist()
        else:
            self.img_paths, self.labels = [], []
            for class_id in sorted(os.listdir(root_dir)):
                if not class_id.isdigit():
                    continue
                class_dir = os.path.join(root_dir, class_id)
                if not os.path.isdir(class_dir):
                    continue
                for img_file in sorted(os.listdir(class_dir)):
                    if not img_file.lower().endswith(('.ppm', '.png', '.jpg', '.jpeg')):
                        continue
                    self.img_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(int(class_id))

    def _parse_semicolon_separated_csv(self, csv_file):
        """Handle the case where all fields are in one column separated by semicolons"""
        with open(csv_file, 'r') as f:
            lines = [line.strip().split(';') for line in f.readlines()]
        
        header = lines[0]
        data = lines[1:]
        return pd.DataFrame(data, columns=header)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            image = cv2.resize(image, self.image_size)

        return image, self.labels[idx]