import os
import torch
from ultralytics import YOLO
from datetime import datetime
import shutil

def main():
    # -----------------------------
    # Paths & Config
    # -----------------------------
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # YAML dataset file
    DATA_YAML = "C:/Users/amanu/Desktop/Projects/TrafficSign/detection/data/YOLOv8_TT100K.yaml"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    # -----------------------------
    # Training Parameters
    # -----------------------------
    EPOCHS = 120
    IMG_SIZE = 640
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = True  # Automatic Mixed Precision

    # Automatically set batch size depending on GPU memory
    if DEVICE == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        BATCH_SIZE = 16 if gpu_mem >= 8 else 8
    else:
        BATCH_SIZE = 2  # Very small for CPU

    print(f"Using device: {DEVICE}, batch size: {BATCH_SIZE}")

    # -----------------------------
    # Verify dataset folders exist
    # -----------------------------
    images_dir = os.path.dirname(DATA_YAML)  # parent folder of the YAML
    train_path = os.path.join(DATA_DIR, "Images", "Train")
    val_path   = os.path.join(DATA_DIR, "Images", "Val")


    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Train or Val folder not found.\nTrain: {train_path}\nVal: {val_path}")
    print("Dataset folders found.")

    # -----------------------------
    # Initialize YOLOv8 Model
    # -----------------------------
    model = YOLO("yolov8s.pt")  # Use yolov8m.pt only if GPU > 8GB

    # -----------------------------
    # Start Training
    # -----------------------------
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        amp=AMP,
        verbose=True,
        workers=2  # safe number for Windows multiprocessing
    )

    end_time = datetime.now()
    print(f"Training finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")

    # -----------------------------
    # Save the trained model
    # -----------------------------
    print("Saving trained model to models directory...")
    runs_dir = os.path.join(BASE_DIR, "runs")
    if os.path.exists(runs_dir):
        for folder in os.listdir(runs_dir):
            src_path = os.path.join(runs_dir, folder)
            dst_path = os.path.join(MODEL_DIR, folder)
            os.makedirs(dst_path, exist_ok=True)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    print("Training complete. Model checkpoints saved.")

if __name__ == "__main__":
    main()
