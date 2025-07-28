import torch
from torch.utils.data import DataLoader
from dataset import TrafficSignDataset  # Your dataset class (needs to accept CSV + dir)
from model import TrafficSignCNN        # Your CNN model definition
from utils import get_class_name         # For class name mapping
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data_dir, batch_size=64, device='cpu'):
    """
    Loads the trained model and evaluates it on the test dataset.
    
    Parameters:
    - model_path: path to saved PyTorch model (.pth)
    - test_data_dir: directory containing test images AND GT-final_test.csv
    - batch_size: how many images to process at once
    - device: 'cpu' or 'cuda'
    """

    # 1. Load the trained model and move to device (CPU/GPU)
    model = TrafficSignCNN(num_classes=43)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # 2. Prepare test dataset and dataloader
    csv_file = f"{test_data_dir}/GT-final_test.csv"  # CSV with labels
    test_dataset = TrafficSignDataset(root_dir=test_data_dir, csv_file=csv_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    # 3. Disable gradient calculation for inference (saves memory and speed)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: get predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class with highest score

            # Collect all predictions and true labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # 5. Compute confusion matrix to understand errors class-wise
    cm = confusion_matrix(all_labels, all_preds)

    # 6. Plot confusion matrix with seaborn for visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Traffic Sign Model")
    parser.add_argument('--model', type=str, default='traffic_sign_model.pth', help='Path to saved model')
    parser.add_argument('--test_dir', type=str, default='data/Test', help='Directory with test images and GT CSV')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')

    args = parser.parse_args()
    evaluate_model(args.model, args.test_dir, args.batch_size, args.device)
