# 🛑 Traffic Sign Recognition using Deep Learning

A deep learning-based project for recognizing German traffic signs using a custom-trained Convolutional Neural Network (CNN) on the [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) dataset. The model supports both image and real-time webcam-based predictions.

## 📌 Features

- ✅ Classification of **43 German traffic signs**
- ✅ Trained on the GTSRB dataset
- ✅ **95%+ accuracy** on the test set
- ✅ **Real-time prediction** using webcam or smartphone camera
- ✅ Utilities for image testing and visualization
- ✅ Modular and clean Python codebase

---

## 📁 Project Structure

├── dataset.py # Custom PyTorch Dataset class for GTSRB
├── train.py # Model training script
├── evaluate.py # Model evaluation script
├── model.py # CNN architecture
├── utils.py # Class mapping & visualization tools
├── predict_sample.py # Predict on a single image
├── real_time_predict.py # Real-time webcam prediction
├── gui_traffic_sign.py # (Optional) GUI-based testing
├── traffic_sign_model.pth # Trained model (if provided)
└── README.md # Project documentation


---

## 🧠 Model Architecture

A simple but effective CNN architecture built using PyTorch. Trained with:
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Data Augmentation: Random crop, flip, normalization

---

## 🎥 Real-Time Demo

> Run the model live using your **webcam or phone camera**.

python real_time_predict.py

If using your phone as a webcam (via apps like DroidCam or Iriun), make sure the correct CAMERA_ID is set in the script.

🧪 Testing
You can test predictions on your own images:

python predict_sample.py

Make sure to change the image path inside the script.

📊 Results
Accuracy: 95%+

Dataset: GTSRB - German Traffic Sign Recognition Benchmark

Training Time: ~5 minutes (on GPU)

🔧 Requirements
Python 3.7+

PyTorch

torchvision

OpenCV

matplotlib

Pillow

Install via pip:
pip install -r requirements.txt


🚀 Future Work
 Integrate YOLOv8 for real-time object detection

 Deploy using Streamlit or Flask

 Mobile or browser-based app using TensorFlow Lite or ONNX

📌 Author
Amanou Allah Nasri
📧 amanullah.nasri@outlook.com
🌐 [LinkedIn](https://www.linkedin.com/in/amanou-allah-nasri-6a5538260/)
📁 [GitHub Repo](https://github.com/AmanouNasri1/TrafficSign)