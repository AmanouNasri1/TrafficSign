import torch
import torchvision.transforms as transforms
from PIL import Image
from model import TrafficSignCNN
from utils import get_class_name, plot_sample_image

# --- Configuration ---
MODEL_PATH = "traffic_sign_model.pth"
NUM_CLASSES = 43
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = TrafficSignCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Load a Sample Image ---
img_path = r"data\Test\Final_Test\Images\00001" \
".ppm"
image = Image.open(img_path)
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# --- Predict ---
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)

# --- Show Result ---
plot_sample_image(input_tensor.squeeze(), label=0, prediction=predicted_class.item())  # Change label to actual label
