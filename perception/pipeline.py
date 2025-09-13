# perception/pipeline.py

import cv2
from pathlib import Path
from ultralytics import YOLO

# Paths
YOLO_MODEL_PATH = Path(__file__).parent.parent / 'runs' / 'detect' / 'train8' / 'weights' / 'best.pt'
TEST_IMAGES_PATH = Path(__file__).parent / 'test_images'  # images for testing

# TT100K class names (replace numbers with readable names if desired)
CLASS_NAMES = {
    0: "pl80", 1: "p6", 2: "ph", 3: "w", 4: "pa", 5: "p27", 6: "i5", 7: "p1",
    8: "il70", 9: "p5", 10: "pm", 11: "p19", 12: "ip", 13: "p11", 14: "p13", 15: "p26",
    16: "i2", 17: "pn", 18: "p10", 19: "p23", 20: "pbp", 21: "p3", 22: "p12",
    23: "pne", 24: "i4", 25: "pb", 26: "pg", 27: "pr", 28: "pl5", 29: "pl10",
    30: "pl15", 31: "pl20", 32: "pl25", 33: "pl30", 34: "pl35", 35: "pl40",
    36: "pl50", 37: "pl60", 38: "pl65", 39: "pl70", 40: "pl90", 41: "pl100",
    42: "pl110", 43: "pl120", 44: "il50", 45: "il60", 46: "il80", 47: "il90",
    48: "il100", 49: "il110"
}

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Get all test images
test_images = list(TEST_IMAGES_PATH.glob("*.jpg"))

# Run detection
for img_path in test_images:
    results = yolo_model(img_path)
    img = cv2.imread(str(img_path))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            label = f"{CLASS_NAMES.get(cls_id, 'Unknown')} ({conf:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show image
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
