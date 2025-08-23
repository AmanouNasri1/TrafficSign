import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import TrafficSignCNN 
from utils import get_class_name

# Settings
MODEL_PATH = "traffic_sign_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA_ID = 0

# Load model
model = TrafficSignCNN(num_classes=43).to(DEVICE)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

# Define transforms (same as during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open webcam stream
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Convert frame (OpenCV uses BGR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)

    # Preprocess frame for model
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        label = get_class_name(predicted_class)

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {label}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Traffic Sign Recognition', frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
