"""
train_tooth_detector.py

Stage 1: trains a YOLO model to detect and number every tooth (32 FDI classes).
Run on a GPU (Google Colab). Run prepare_data.py and prepare_yolo_teeth.py first.
"""
import sys
from ultralytics import YOLO

# Where to save results. On Colab we pass a Google Drive folder so that
# the trained model is not lost if the session disconnects.
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "runs"

model = YOLO("yolo11m.pt")  # pretrained medium-size YOLO11 model

model.train(
    data="yolo_teeth/data.yaml",
    epochs=100,
    imgsz=1024,
    batch=8,
    patience=20,     # stop early if no improvement for 20 epochs
    fliplr=0.0,      # NO left-right flip: it would swap tooth numbers
    project=PROJECT,
    name="teeth",
    seed=42,
)