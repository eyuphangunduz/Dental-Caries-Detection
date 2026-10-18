"""
check_prediction.py

Prints the ground-truth (dentist) labels of one test X-ray,
so predict.py's output can be compared with the truth.

Usage: python check_prediction.py [index]   (default index 0 = same image as predict.py)
"""
import json
import os
import sys

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
with open(os.path.join("data", "test.json"), encoding="utf-8") as f:
    image = json.load(f)[index]

print("Image:", image["path"])
names = {"caries": "Çürük", "deep_caries": "Derin çürük", "other": "Diğer (gömülü/lezyon)"}
for label, name in names.items():
    teeth = [t["fdi"] for t in image["teeth"] if t["label"] == label]
    print(f"{name}: {teeth if teeth else '-'}")