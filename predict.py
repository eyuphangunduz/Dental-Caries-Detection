"""
predict.py

Runs the full system on one panoramic X-ray:
  1) YOLO finds and numbers every tooth            (models/best.pt)
  2) every tooth is cropped (same 10% margin as in training)
  3) the fine-tuned ResNet-50 turns each crop into 2048 features and also
     gives its own 3-class prediction           (models/resnet50_finetuned.pt)
  4) scaler + feature selection + SVM classify each tooth  (models/best_model.joblib)
  5) if models/decision.json exists (made by tune_threshold.py), the CNN and SVM
     probabilities are combined (ensemble) and a threshold is applied,
     otherwise the SVM decision is used
  6) carious teeth are drawn on the X-ray with their FDI number and label

Usage:  python predict.py [path/to/xray.png] [output.png]
        Without arguments the first image of the test split (data/test.json) is used.

Research project - not a medical diagnosis tool.
"""
import json
import os
import sys

import joblib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

MODEL_DIR = "models"
PAD = 0.10                 # same margin as crop_teeth.py
IMG_SIZE = (320, 224)      # same input size as finetune_cnn.py
FDI_CLASSES = [q * 10 + t for q in range(1, 5) for t in range(1, 9)]
LABEL_TEXT = {0: "Sağlıklı", 1: "Çürük", 2: "Derin çürük"}
COLORS = {1: (255, 165, 0), 2: (230, 30, 30)}   # orange = caries, red = deep caries

TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_models():
    """Loads the trained models once (they are reused for every X-ray)."""
    yolo = YOLO(os.path.join(MODEL_DIR, "best.pt"))

    cnn = models.resnet50(weights=None)
    cnn.fc = torch.nn.Linear(cnn.fc.in_features, 3)
    cnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet50_finetuned.pt"),
                                   map_location="cpu"))
    cnn.eval()
    backbone = torch.nn.Sequential(*list(cnn.children())[:-1])  # everything except fc

    bundle = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))

    decision = None
    decision_path = os.path.join(MODEL_DIR, "decision.json")
    if os.path.exists(decision_path):
        with open(decision_path, encoding="utf-8") as f:
            decision = json.load(f)
    return yolo, cnn, backbone, bundle, decision


def crop(img, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - PAD * w), max(0, y1 - PAD * h)
    x2, y2 = min(img.width, x2 + PAD * w), min(img.height, y2 + PAD * h)
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def select_features(X, bundle):
    """Applies the same scaling and feature selection as in run_experiments.py."""
    X = bundle["scaler"].transform(X)
    kind, value = bundle["selector"]
    if kind == "pca":
        return bundle["pca"].transform(X)[:, :value]
    if kind == "idx":
        return X[:, value]
    return X


def tooth_scores(crops, cnn, backbone, bundle):
    """Returns (CNN probabilities, SVM probabilities, SVM features) for a list of tooth crops."""
    batch = torch.stack([TRANSFORM(c) for c in crops])
    with torch.no_grad():
        features = backbone(batch).flatten(1)
        logits = cnn.fc(features)
    selected = select_features(features.numpy(), bundle)
    cnn_prob = softmax(logits.numpy())
    svm_prob = softmax(bundle["classifier"].decision_function(selected))
    return cnn_prob, svm_prob, selected


def decide(cnn_prob, svm_prob, decision):
    """Ensemble + threshold (bias) rule chosen by tune_threshold.py."""
    w = decision["cnn_weight"]
    prob = w * cnn_prob + (1 - w) * svm_prob
    return np.argmax(np.log(prob + 1e-9) + np.array(decision["bias"]), axis=1)


def get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def analyze(image, loaded_models):
    """
    image: PIL image of a panoramic X-ray
    Returns (annotated image, findings, number of detected teeth).
    findings = list of (fdi_number, class_id) for carious teeth.
    """
    yolo, cnn, backbone, bundle, decision = loaded_models
    rgb = image.convert("RGB")
    gray = image.convert("L")

    result = yolo.predict(rgb, imgsz=1024, conf=0.25, agnostic_nms=True, verbose=False)[0]
    boxes = result.boxes.xyxy.tolist()
    fdis = [FDI_CLASSES[int(c)] for c in result.boxes.cls.tolist()]

    findings = []
    draw = ImageDraw.Draw(rgb)
    font = get_font(max(24, rgb.width // 70))

    if boxes:
        cnn_prob, svm_prob, selected = tooth_scores([crop(gray, b) for b in boxes],
                                                    cnn, backbone, bundle)
        if decision:
            preds = decide(cnn_prob, svm_prob, decision)
        else:
            preds = bundle["classifier"].predict(selected)

        for box, fdi, pred in zip(boxes, fdis, preds):
            pred = int(pred)
            if pred == 0:
                continue  # healthy teeth are not marked
            findings.append((fdi, pred))
            color = COLORS[pred]
            draw.rectangle(box, outline=color, width=max(3, rgb.width // 600))
            text = f"{fdi}: {LABEL_TEXT[pred]}"
            tx, ty = box[0], max(0, box[1] - font.size - 8)
            tb = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle((tb[0] - 4, tb[1] - 4, tb[2] + 4, tb[3] + 4), fill=color)
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    if not findings:
        text = "Çürük tespit edilmedi"
        width = draw.textbbox((0, 0), text, font=font)[2]
        pos = (rgb.width - width - 30, 30)   # top-right corner
        tb = draw.textbbox(pos, text, font=font)
        draw.rectangle((tb[0] - 8, tb[1] - 8, tb[2] + 8, tb[3] + 8), fill=(40, 160, 70))
        draw.text(pos, text, fill=(255, 255, 255), font=font)

    return rgb, sorted(findings), len(boxes)


def summary_text(findings, n_teeth):
    if n_teeth == 0:
        return "Röntgende diş bulunamadı. Lütfen bir panoramik diş röntgeni yükleyin."
    caries = [fdi for fdi, c in findings if c == 1]
    deep = [fdi for fdi, c in findings if c == 2]
    lines = [f"{n_teeth} diş bulundu."]
    if not findings:
        lines.append("Çürük tespit edilmedi.")
    else:
        if deep:
            lines.append(f"Derin çürük ({len(deep)} diş): " + ", ".join(map(str, deep)))
        if caries:
            lines.append(f"Çürük ({len(caries)} diş): " + ", ".join(map(str, caries)))
    lines.append("\nNot: Bu bir araştırma projesidir, tıbbi teşhis yerine geçmez.")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        with open(os.path.join("data", "test.json"), encoding="utf-8") as f:
            path = json.load(f)[0]["path"]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(path))[0] + "_result.png"

    print("Loading models...")
    loaded = load_models()
    print("Decision rule:", "ensemble + threshold" if loaded[4] else "SVM only")
    print("Analyzing:", path)
    annotated, findings, n_teeth = analyze(Image.open(path), loaded)
    annotated.save(out_path)

    print()
    print(summary_text(findings, n_teeth))
    print("\nResult image saved to", out_path)


if __name__ == "__main__":
    main()