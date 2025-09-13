import os

base = "C:/Users/amanu/Desktop/Projects/TrafficSign/detection/data/Images"
print("Train exists:", os.path.exists(os.path.join(base, "Train")))
print("Val exists:", os.path.exists(os.path.join(base, "Val")))