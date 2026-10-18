"""
extract_features.py

Stage 2: turns every tooth crop into a 2048-number feature vector using a
ResNet-50 as a feature extractor (its last layer is removed).

Two modes:
  python extract_features.py
      -> ImageNet-pretrained ResNet-50 (no training on X-rays), 224x224 input.
         Output folder: features/
  python extract_features.py --weights models/resnet50_finetuned.pt --out features_ft
      -> ResNet-50 fine-tuned on tooth crops (finetune_cnn.py), 320x224 input.

Output: <out>/<split>_X.npy    -> feature matrix (n_teeth x 2048)
        <out>/<split>_y.npy    -> labels (0 = healthy, 1 = caries, 2 = deep caries)
        <out>/<split>_meta.csv -> which crop each row belongs to
Run on a GPU (Colab) after crop_teeth.py.
"""
import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

CROPS = "crops"
BATCH = 64
LABELS = {"healthy": 0, "caries": 1, "deep_caries": 2}

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=None, help="fine-tuned weights from finetune_cnn.py")
parser.add_argument("--out", default="features", help="output folder")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
if args.weights:
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    img_size = (320, 224)   # same size as used for fine-tuning
else:
    img_size = (224, 224)
model.fc = torch.nn.Identity()  # remove the classification layer -> 2048 features
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.Grayscale(num_output_channels=3),   # X-ray (1 channel) -> 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    os.makedirs(args.out, exist_ok=True)
    print("Device:", device, "| weights:", args.weights or "ImageNet", "| input size:", img_size)

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

        np.save(os.path.join(args.out, f"{split}_X.npy"), X)
        np.save(os.path.join(args.out, f"{split}_y.npy"), y)
        with open(os.path.join(args.out, f"{split}_meta.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"{split}: X shape = {X.shape}, labels = {np.bincount(y, minlength=3).tolist()}")

    print(f"\nSaved to '{args.out}/'")


if __name__ == "__main__":
    main()