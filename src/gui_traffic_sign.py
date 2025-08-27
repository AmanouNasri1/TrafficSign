import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from model import TrafficSignCNN
from utils import get_class_name

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN(num_classes=43).to(DEVICE)
model.load_state_dict(torch.load("traffic_sign_model.pth", map_location=DEVICE, weights_only=True))
model.eval()

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize camera
cap = cv2.VideoCapture(0)

def predict_frame():
    ret, frame = cap.read()
    if ret:
        # Convert frame to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Preprocess and predict
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = get_class_name(predicted.item())

        # Update GUI label with prediction
        label_var.set(f"Prediction: {label}")

        # Convert PIL image to ImageTk format
        imgtk = ImageTk.PhotoImage(image=pil_img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Schedule the function to run again after 10 ms
    video_label.after(10, predict_frame)

# Setup Tkinter window
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.minsize(height=500,width=500)

label_var = tk.StringVar()
label_var.set("Prediction: ")

video_label = Label(root)
video_label.pack()

pred_label = Label(root, textvariable=label_var, font=("Helvetica", 16))
pred_label.pack()

start_button = Button(root, text="Start Prediction", command=predict_frame)
start_button.pack()

# Run the Tkinter event loop
root.mainloop()

# Release camera when window closes
cap.release()
