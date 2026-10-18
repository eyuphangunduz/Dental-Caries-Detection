"""
finetune_cnn.py

Stage 2 (improved features): fine-tunes an ImageNet-pretrained ResNet-50 on the
tooth crops (healthy / caries / deep caries), so that the network learns X-ray
and caries specific features instead of generic photo features.

- Class imbalance: weighted cross-entropy (rare classes get larger weights)
- Augmentation: small rotation, horizontal flip, brightness/contrast changes
- The epoch with the best validation macro F1 is kept.
- At the end the fine-tuned CNN itself is evaluated on the test set
  (end-to-end baseline for the paper).

The saved weights are then used by extract_features.py (--weights) so that the
feature selection experiments can be repeated on fine-tuned features.

Usage:  python finetune_cnn.py OUTPUT_WEIGHTS_PATH
Run on a GPU (Colab) after crop_teeth.py.
"""
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CROPS = "crops"
OUT_WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "resnet50_finetuned.pt")
EPOCHS = 15
BATCH = 64
LR = 1e-4
SEED = 42
IMG_SIZE = (320, 224)  # (height, width): teeth are taller than wide
LABELS = {"healthy": 0, "caries": 1, "deep_caries": 2}
CLASS_NAMES = ["Healthy", "Caries", "Deep caries"]

NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    NORMALIZE,
])
EVAL_TF = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    NORMALIZE,
])


class ToothCrops(Dataset):
    def __init__(self, split, transform):
        with open(os.path.join(CROPS, f"{split}.csv"), encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(os.path.join(CROPS, row["path"]))
        return self.transform(image), LABELS[row["label"]]


def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = ToothCrops("train", TRAIN_TF)
    val_ds = ToothCrops("val", EVAL_TF)
    test_ds = ToothCrops("test", EVAL_TF)

    counts = np.bincount([LABELS[r["label"]] for r in train_ds.rows], minlength=3)
    class_weights = torch.tensor(len(train_ds) / (3 * counts), dtype=torch.float32, device=device)
    print("Train counts:", counts.tolist(),
          "| class weights:", class_weights.cpu().numpy().round(2).tolist())

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=8, pin_memory=True)

    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    os.makedirs(os.path.dirname(OUT_WEIGHTS) or ".", exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
        scheduler.step()

        y_val, p_val = predict(model, val_dl, device)
        val_f1 = f1_score(y_val, p_val, average="macro")
        saved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), OUT_WEIGHTS)
            saved = "  <- best, saved"
        print(f"epoch {epoch:2d}/{EPOCHS}  loss={total_loss / len(train_ds):.3f}  "
              f"val macro-F1={val_f1:.3f}  ({time.time() - start:.0f}s){saved}")

    model.load_state_dict(torch.load(OUT_WEIGHTS, map_location=device))
    y_test, p_test = predict(model, test_dl, device)
    print(f"\nBest validation macro-F1: {best_f1:.3f}")
    print("Fine-tuned CNN (end-to-end) on the TEST set:")
    print(classification_report(y_test, p_test, target_names=CLASS_NAMES, digits=3))
    print("Weights saved to", OUT_WEIGHTS)


if __name__ == "__main__":
    main()