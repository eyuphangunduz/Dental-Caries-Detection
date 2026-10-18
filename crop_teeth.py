"""
crop_teeth.py

Stage 2 data: uses the trained YOLO tooth detector to find every tooth on the
caries X-rays (data/train.json, val.json, test.json), crops each tooth and
labels it as healthy / caries / deep_caries.

Labelling rule for each detected tooth:
  - overlaps a caries or deep caries annotation (IoU >= 0.3) -> that label
  - overlaps an "other" annotation (impacted, periapical)    -> skipped
  - no overlap with any annotation (IoU < 0.1)                -> healthy
  - unclear overlap (0.1 - 0.3)                               -> skipped
Train split only: carious teeth missed by the detector are added with their
ground-truth box, so no positive training examples are lost.
Val/test: missed carious teeth are only counted (reported as detector misses).

Usage:  python crop_teeth.py path/to/best.pt
Output: crops/<split>/<label>/*.png and crops/<split>.csv
"""
import csv
import json
import os
import sys
from collections import Counter

from PIL import Image
from ultralytics import YOLO

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "best.pt")
OUT = "crops"
PAD = 0.10          # add 10% margin around each tooth box
IOU_MATCH = 0.3
IOU_HEALTHY = 0.1
LABELS = ("healthy", "caries", "deep_caries")
FDI_CLASSES = [q * 10 + t for q in range(1, 5) for t in range(1, 9)]


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def crop(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - PAD * w), max(0, y1 - PAD * h)
    x2, y2 = min(img.width, x2 + PAD * w), min(img.height, y2 + PAD * h)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def main():
    model = YOLO(MODEL)

    for split in ("train", "val", "test"):
        with open(os.path.join("data", f"{split}.json"), encoding="utf-8") as f:
            images = json.load(f)
        for label in LABELS:
            os.makedirs(os.path.join(OUT, split, label), exist_ok=True)

        rows, missed = [], 0

        def save(img, box, label, fdi, file_name, source):
            stem = os.path.splitext(file_name)[0]
            name = f"{stem}_{fdi}_{len(rows)}.png"
            crop(img, box).save(os.path.join(OUT, split, label, name))
            rows.append([f"{split}/{label}/{name}", label, fdi, file_name, source])

        for n, im in enumerate(images, 1):
            img = Image.open(im["path"]).convert("L")
            gts = [(xywh_to_xyxy(t["bbox"]), t["label"], t["fdi"]) for t in im["teeth"]]
            matched = set()

            result = model.predict(im["path"], imgsz=1024, conf=0.25, verbose=False)[0]
            for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
                fdi = FDI_CLASSES[int(cls)]
                ious = [iou(box, g[0]) for g in gts]
                best = max(range(len(ious)), key=ious.__getitem__) if ious else None
                best_iou = ious[best] if ious else 0.0

                if best_iou >= IOU_MATCH:
                    matched.add(best)
                    label = gts[best][1]
                    if label == "other":
                        continue
                elif best_iou < IOU_HEALTHY:
                    label = "healthy"
                else:
                    continue
                save(img, box, label, fdi, im["file_name"], "detected")

            # Carious teeth the detector did not find
            for i, (gbox, label, fdi) in enumerate(gts):
                if i in matched or label == "other":
                    continue
                if split == "train":
                    save(img, gbox, label, fdi, im["file_name"], "ground_truth")
                else:
                    missed += 1

            if n % 50 == 0:
                print(f"  {split}: {n}/{len(images)} images")

        with open(os.path.join(OUT, f"{split}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "fdi", "source_image", "source"])
            writer.writerows(rows)

        c = Counter(r[1] for r in rows)
        print(f"{split}: healthy={c['healthy']}  caries={c['caries']}  "
              f"deep_caries={c['deep_caries']}  missed_by_detector={missed}")

    print(f"\nSaved to '{OUT}/'")


if __name__ == "__main__":
    main()