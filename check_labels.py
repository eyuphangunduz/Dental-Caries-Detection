import json
import os
from collections import Counter

BASE = os.path.join("DENTEX", "DENTEX")

TRAIN = os.path.join(BASE, "training_data", "quadrant-enumeration-disease",
                     "train_quadrant_enumeration_disease.json")
VAL = os.path.join(BASE, "validation_triple.json")
TEST = os.path.join(BASE, "disease", "label", "test_0.json")


def short(obj, limit=600):
    text = json.dumps(obj, ensure_ascii=False)
    return text[:limit] + (" ..." if len(text) > limit else "")


for name, path in [("TRAIN", TRAIN), ("VALIDATION", VAL), ("TEST (test_0)", TEST)]:
    print("=" * 60)
    print(name, "->", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))
        for key, value in data.items():
            if key.startswith("categories"):
                print(f"\n{key}:", short(value, 1500))
        if "annotations" in data:
            print("\nNumber of annotations:", len(data["annotations"]))
            print("First annotation:", short(data["annotations"][0]))
            if "category_id_3" in data["annotations"][0]:
                counts = Counter(a["category_id_3"] for a in data["annotations"])
                print("Disease label counts (category_id_3):", dict(counts))
        if "images" in data:
            print("Number of images:", len(data["images"]))
            print("First image:", short(data["images"][0]))
        if "shapes" in data:
            print("\nNumber of shapes:", len(data["shapes"]))
            print("First shape:", short(data["shapes"][0]))
    else:
        print("Type:", type(data).__name__, "| content:", short(data))
    print()