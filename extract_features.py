"""
extract_features.py

Stage 2: turns every tooth crop into a 2048-number feature vector using a
ResNet-50 pretrained on ImageNet (transfer learning: the network is NOT
trained, it is only used as a feature extractor).
Run on a GPU (Colab) after crop_teeth.py.

Output: features/<split>_X.npy    -> feature matrix (n_teeth x 2048)
        features/<split>_y.npy    -> labels (0 = healthy, 1 = caries, 2 = deep caries)
        features/<split>_meta.csv -> which crop each row belongs to
"""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

CROPS = "crops"
OUT = "features"
BATCH = 64
LABELS = {"healthy": 0, "caries": 1, "deep_caries": 2}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Pretrained ResNet-50 without its last (classification) layer
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # X-ray (1 channel) -> 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Device:", device)

    for split in ("train", "val", "test"):
        with open(os.path.join(CROPS, f"{split}.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        features = []
        for i in range(0, len(rows), BATCH):
            batch = torch.stack([
                transform(Image.open(os.path.join(CROPS, r["path"])))
                for r in rows[i:i + BATCH]
            ]).to(device)
            with torch.no_grad():
                features.append(model(batch).cpu().numpy())

        X = np.concatenate(features).astype(np.float32)
        y = np.array([LABELS[r["label"]] for r in rows])

        np.save(os.path.join(OUT, f"{split}_X.npy"), X)
        np.save(os.path.join(OUT, f"{split}_y.npy"), y)
        with open(os.path.join(OUT, f"{split}_meta.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"{split}: X shape = {X.shape}, labels = {np.bincount(y, minlength=3).tolist()}")

    print(f"\nSaved to '{OUT}/'")


if __name__ == "__main__":
    main()