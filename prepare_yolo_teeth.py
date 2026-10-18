"""
prepare_yolo_teeth.py

Stage 1 data: converts the DENTEX "quadrant_enumeration" set (634 X-rays,
every tooth labelled with its number) into YOLO format for tooth detection.

Each tooth gets one of 32 classes = its FDI number (11..18, 21..28, 31..38, 41..48).

Images that also appear in our caries val/test splits are removed, so the
tooth detector never sees those X-rays (prevents data leakage).

Output: yolo_teeth/ folder + yolo_teeth/data.yaml
Run prepare_data.py first.
"""
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.15
BASE = os.path.join("DENTEX", "DENTEX")
ENUM_JSON = os.path.join(BASE, "training_data", "quadrant_enumeration",
                         "train_quadrant_enumeration.json")
ENUM_IMG_DIR = os.path.join(BASE, "training_data", "quadrant_enumeration", "xrays")
OUT = "yolo_teeth"

FDI_CLASSES = [q * 10 + t for q in range(1, 5) for t in range(1, 9)]
CLASS_INDEX = {fdi: i for i, fdi in enumerate(FDI_CLASSES)}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    # 1) Fingerprints of the caries val/test images (must not be used here)
    protected = set()
    for split in ("val", "test"):
        with open(os.path.join("data", f"{split}.json"), encoding="utf-8") as f:
            for im in json.load(f):
                protected.add(md5(im["path"]))

    # 2) Read tooth labels
    with open(ENUM_JSON, encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}

    labels = defaultdict(list)
    for a in data["annotations"]:
        img = images[a["image_id"]]
        W, H = img["width"], img["height"]
        x, y, w, h = a["bbox"]
        fdi = (a["category_id_1"] + 1) * 10 + (a["category_id_2"] + 1)
        labels[a["image_id"]].append(
            f"{CLASS_INDEX[fdi]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
            f"{w / W:.6f} {h / H:.6f}"
        )

    # 3) Keep usable images
    kept, removed = [], 0
    for img in images.values():
        path = os.path.join(ENUM_IMG_DIR, img["file_name"])
        if not os.path.exists(path):
            print("WARNING: image not found:", path)
            continue
        if md5(path) in protected:
            removed += 1
            continue
        kept.append(img)

    # 4) Split and write YOLO folders
    random.Random(SEED).shuffle(kept)
    n_val = round(len(kept) * VAL_RATIO)
    splits = {"val": kept[:n_val], "train": kept[n_val:]}

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for split, items in splits.items():
        os.makedirs(os.path.join(OUT, "images", split))
        os.makedirs(os.path.join(OUT, "labels", split))
        for img in items:
            src = os.path.join(ENUM_IMG_DIR, img["file_name"])
            shutil.copy(src, os.path.join(OUT, "images", split, img["file_name"]))
            stem = os.path.splitext(img["file_name"])[0]
            with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(labels[img["id"]]))

    # 5) data.yaml for YOLO
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, fdi in enumerate(FDI_CLASSES):
            f.write(f"  {i}: '{fdi}'\n")

    n_teeth = sum(len(labels[im["id"]]) for im in kept)
    print(f"Images used: {len(kept)}  (removed {removed} overlapping with caries val/test)")
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Teeth labelled: {n_teeth}")
    print(f"Saved to '{OUT}/'")


if __name__ == "__main__":
    main()