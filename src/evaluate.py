import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import TrafficSignCNN
from dataset import TrafficSignDataset

# ===== Paths =====
TEST_CSV = r"C:\Users\amanu\Desktop\Projects\TrafficSign\data\Test\GT-final_test.csv"
TEST_IMAGES_DIR = r"C:\Users\amanu\Desktop\Projects\TrafficSign\data\Test\Final_Test\Images"
MODEL_PATH = "traffic_sign_model.pth"
NUM_CLASSES = 43

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Transforms =====
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ===== Dataset & DataLoader =====
test_dataset = TrafficSignDataset(root_dir=TEST_IMAGES_DIR, csv_file=TEST_CSV, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===== Model =====
model = TrafficSignCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===== Evaluation =====
criterion = nn.CrossEntropyLoss()
total_loss, correct, total = 0, 0, 0
all_labels, all_preds = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# ===== Results =====
avg_loss = total_loss / total
accuracy = 100 * correct / total

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# ===== Confusion Matrix =====
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
