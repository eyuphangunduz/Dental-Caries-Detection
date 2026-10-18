"""
prepare_data.py

Merges the DENTEX training (705) and validation (50) sets, which share the
same label format, extracts tooth-level diagnoses and splits the images into
train / val / test (70% / 15% / 15%).

Output: data/train.json, data/val.json, data/test.json
"""
import json
import os
import random
from collections import Counter, defaultdict

SEED = 42
BASE = os.path.join("DENTEX", "DENTEX")
OUT_DIR = "data"

SOURCES = [
    (
        os.path.join(BASE, "training_data", "quadrant-enumeration-disease",
                     "train_quadrant_enumeration_disease.json"),
        os.path.join(BASE, "training_data", "quadrant-enumeration-disease", "xrays"),
    ),
    (
        os.path.join(BASE, "validation_triple.json"),
        os.path.join(BASE, "validation_data", "quadrant_enumeration_disease", "xrays"),
    ),
]

# DENTEX category_id_3 -> our label.
# "other" (impacted, periapical lesion) is kept only so that these teeth
# are NOT counted as healthy later.
DISEASE_MAP = {0: "other", 1: "caries", 2: "other", 3: "deep_caries"}
PRIORITY = {"deep_caries": 3, "caries": 2, "other": 1}


def fdi_number(ann):
    """Quadrant (0-3) and tooth (0-7) ids -> FDI number, e.g. 3,7 -> 48."""
    return (ann["category_id_1"] + 1) * 10 + (ann["category_id_2"] + 1)


def union_box(boxes):
    """Merge several [x, y, w, h] boxes into one box that covers all of them."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return [x1, y1, x2 - x1, y2 - y1]


def load_source(json_path, image_dir):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    images = {}
    for img in data["images"]:
        path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(path):
            print("WARNING: image not found, skipped:", path)
            continue
        images[img["id"]] = {
            "file_name": img["file_name"],
            "path": path,
            "width": img["width"],
            "height": img["height"],
            "raw": defaultdict(list),  # fdi -> list of annotations
        }

    for ann in data["annotations"]:
        if ann["image_id"] in images:
            images[ann["image_id"]]["raw"][fdi_number(ann)].append(ann)

    # One entry per tooth. If a tooth has several diagnoses, the most severe
    # one wins: deep_caries > caries > other.
    result = []
    for img in images.values():
        teeth = []
        for fdi, anns in img.pop("raw").items():
            labels = [DISEASE_MAP[a["category_id_3"]] for a in anns]
            teeth.append({
                "fdi": fdi,
                "label": max(labels, key=lambda l: PRIORITY[l]),
                "bbox": union_box([a["bbox"] for a in anns]),
                "polygon": anns[0]["segmentation"][0],
            })
        img["teeth"] = sorted(teeth, key=lambda t: t["fdi"])
        result.append(img)
    return result


def split(images):
    """Stratified split so that images with deep caries are spread evenly."""
    rng = random.Random(SEED)
    with_deep = [im for im in images if any(t["label"] == "deep_caries" for t in im["teeth"])]
    without_deep = [im for im in images if im not in with_deep]

    parts = {"train": [], "val": [], "test": []}
    for group in (with_deep, without_deep):
        rng.shuffle(group)
        n = len(group)
        n_train, n_val = round(n * 0.70), round(n * 0.15)
        parts["train"] += group[:n_train]
        parts["val"] += group[n_train:n_train + n_val]
        parts["test"] += group[n_train + n_val:]
    return parts


def main():
    images = []
    for json_path, image_dir in SOURCES:
        loaded = load_source(json_path, image_dir)
        print(f"Loaded {len(loaded)} images from {os.path.basename(json_path)}")
        images += loaded

    parts = split(images)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'split':6s} {'images':>7s} {'caries':>7s} {'deep':>6s} {'other':>6s}")
    for name, items in parts.items():
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=1)
        c = Counter(t["label"] for im in items for t in im["teeth"])
        print(f"{name:6s} {len(items):7d} {c['caries']:7d} {c['deep_caries']:6d} {c['other']:6d}")

    print(f"\nSaved to '{OUT_DIR}/' folder.")


if __name__ == "__main__":
    main()