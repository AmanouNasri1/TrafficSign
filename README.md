# Traffic Sign Detection and Recognition

This project implements a traffic sign detection and recognition pipeline using **YOLOv8** for detection and a CNN-based classifier for the GTSRB dataset. It also supports predictions using the TT100K dataset for detection.

---

## Project Structure

repo/
│
├─ classification/ # Traffic sign classification
│ ├─ data/ # Images for training/testing
│ ├─ src/ # Python code (model, utils, etc.)
│ └─ traffic_sign_model.pth # Trained classification model
│
├─ detection/ # YOLO detection
│ ├─ runs/ # YOLO training & validation outputs
│ ├─ models/ # YOLO models, args.yaml
│ └─ data/ # TT100K images & labels
│
├─ perception/ # Detection pipeline
│ ├─ pipeline.py # Run detection and classification
│ └─ test_images/ # Images for testing the pipeline
│
├─ .venv/ # Python virtual environment
└─ README.md

yaml
Copy code

---

## Datasets

### Classification
- **GTSRB dataset**: 43 traffic sign classes (German signs).  
  Classes example: `Speed limit 20 km/h`, `Stop`, `Yield`, `No entry`, etc.

### Detection
- **TT100K dataset**: 50 classes of Chinese traffic signs.  
  Example classes: `pl80`, `p6`, `ph`, `w`, `pa`, `i5`, `il90`, etc.

---

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd TrafficSign
Create a virtual environment:

bash
Copy code
python -m venv .venv
Activate the environment:

bash
Copy code
# Windows
.venv\Scripts\activate

# Linux / MacOS
source .venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Requirements include: torch, torchvision, ultralytics, opencv-python, matplotlib, etc.

Usage
Run the Perception Pipeline
The pipeline detects traffic signs in images and classifies them (optional for GTSRB):

bash
Copy code
python perception/pipeline.py
Test images should be placed in perception/test_images/.

YOLO detection model is loaded from runs/detect/train8/weights/best.pt.

Classification model is loaded from classification/traffic_sign_model.pth.

Folder Descriptions
classification/src: Contains the CNN model definition and utility functions.

classification/data: Training/testing images for classification.

detection: Contains YOLO dataset, models, and results.

perception/pipeline.py: Runs the full detection (and optional classification) pipeline.

perception/test_images: Images to run inference on.

Notes
YOLO weights can be large (~50–100 MB). If not included in the repository, you can retrain the model using the detection dataset:

bash
Copy code
# Example YOLO training command
yolo train model=yolov8n.pt data=detection/data/YOLOv8_TT100K.yaml epochs=120
Classification model (traffic_sign_model.pth) must match the architecture defined in classification/src/model.py.

Known Issues
Currently, the pipeline is adapted for TT100K detection only. Classification can be optionally included for GTSRB images.

Predictions may vary depending on the training quality and dataset size.

References
YOLOv8 Documentation

GTSRB Dataset

TT100K Dataset

📌 Authors
Project developed by Amanou Allah Nasri (adapted for personal use and experimentation).
📧 amanullah.nasri@outlook.com
🌐 [LinkedIn](https://www.linkedin.com/in/amanou-allah-nasri-6a5538260/)
📁 [GitHub Repo](https://github.com/AmanouNasri1/TrafficSign)