"""
tune_threshold.py

Improves the caries decision WITHOUT retraining any model:
  1) Ensemble: combines the class probabilities of the fine-tuned CNN (its own
     3-class output) and of the PCA + SVM classifier:
         p = w * p_cnn + (1 - w) * p_svm
  2) Threshold (class bias): a carious class must beat "healthy" by a margin,
         prediction = argmax( log(p) + [0, b_caries, b_deep] )
     Negative biases -> the system says "caries" less often -> fewer false alarms.

w, b_caries and b_deep are chosen on the VALIDATION set only (best macro F1).
The TEST set is evaluated once at the end and compared with the old rules.

Tooth crops and labels are produced exactly like crop_teeth.py (same YOLO
model, same matching rules), so the numbers are comparable with earlier results.

Runs on CPU (about 15-25 minutes the first time; scores are cached in
tuning_cache.npz so later runs take seconds).
Output: models/decision.json (used automatically by predict.py and app.py)
        results_tuning/comparison.csv, results_tuning/threshold_curve.png
"""
import json
import os
import time
from itertools import product

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from crop_teeth import IOU_HEALTHY, IOU_MATCH, iou, xywh_to_xyxy
from predict import MODEL_DIR, crop, decide, load_models, tooth_scores

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CACHE = "tuning_cache.npz"
OUT = "results_tuning"
LABELS = {"healthy": 0, "caries": 1, "deep_caries": 2}
CLASS_NAMES = ["Healthy", "Caries", "Deep caries"]
CNN_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
BIAS_GRID = np.round(np.arange(-2.0, 2.01, 0.25), 2)


def collect(split, yolo, cnn, backbone, bundle):
    """Detects, crops and labels every tooth of a split; returns CNN and SVM probabilities."""
    with open(os.path.join("data", f"{split}.json"), encoding="utf-8") as f:
        images = json.load(f)

    cnn_all, svm_all, labels = [], [], []
    start = time.time()
    for n, im in enumerate(images, 1):
        img = Image.open(im["path"]).convert("L")
        gts = [(xywh_to_xyxy(t["bbox"]), t["label"]) for t in im["teeth"]]
        result = yolo.predict(im["path"], imgsz=1024, conf=0.25,
                              agnostic_nms=True, verbose=False)[0]
        boxes = result.boxes.xyxy.tolist()

        # Same matching rules as crop_teeth.py
        info = []
        for box in boxes:
            ious = [iou(box, g[0]) for g in gts]
            best = int(np.argmax(ious)) if ious else None
            info.append((best, max(ious) if ious else 0.0))
        assigned = {}
        for d, (best, best_iou) in enumerate(info):
            if best_iou >= IOU_MATCH and (best not in assigned or best_iou > assigned[best][1]):
                assigned[best] = (d, best_iou)

        kept_crops = []
        for d, (box, (best, best_iou)) in enumerate(zip(boxes, info)):
            if best_iou >= IOU_MATCH:
                if assigned[best][0] != d:
                    continue
                label = gts[best][1]
                if label == "other":
                    continue
            elif best_iou < IOU_HEALTHY:
                label = "healthy"
            else:
                continue
            kept_crops.append(crop(img, box))
            labels.append(LABELS[label])

        if kept_crops:
            cnn_prob, svm_prob, _ = tooth_scores(kept_crops, cnn, backbone, bundle)
            cnn_all.append(cnn_prob)
            svm_all.append(svm_prob)

        if n % 10 == 0 or n == len(images):
            print(f"  {split}: {n}/{len(images)} images  ({time.time() - start:.0f}s)")

    return np.concatenate(cnn_all), np.concatenate(svm_all), np.array(labels)


def metrics(y, pred):
    """Macro F1, per-class F1 and 'any caries vs healthy' precision / recall."""
    per_class = f1_score(y, pred, average=None, labels=[0, 1, 2])
    y_any, p_any = (y > 0).astype(int), (pred > 0).astype(int)
    return dict(
        macro_f1=round(f1_score(y, pred, average="macro"), 4),
        f1_healthy=round(per_class[0], 4),
        f1_caries=round(per_class[1], 4),
        f1_deep_caries=round(per_class[2], 4),
        any_caries_precision=round(precision_score(y_any, p_any, zero_division=0), 4),
        any_caries_recall=round(recall_score(y_any, p_any, zero_division=0), 4),
    )


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1) Probabilities for every tooth of val and test (cached)
    if os.path.exists(CACHE):
        print("Loading cached scores from", CACHE)
        c = np.load(CACHE)
        cv, sv, yv, ct, st, yt = (c[k] for k in ("cv", "sv", "yv", "ct", "st", "yt"))
    else:
        print("Loading models...")
        yolo, cnn, backbone, bundle, _ = load_models()
        print("Scoring validation set...")
        cv, sv, yv = collect("val", yolo, cnn, backbone, bundle)
        print("Scoring test set...")
        ct, st, yt = collect("test", yolo, cnn, backbone, bundle)
        np.savez(CACHE, cv=cv, sv=sv, yv=yv, ct=ct, st=st, yt=yt)
    print(f"Val teeth: {np.bincount(yv, minlength=3).tolist()}  "
          f"Test teeth: {np.bincount(yt, minlength=3).tolist()}\n")

    # 2) Choose ensemble weight and biases on the VALIDATION set
    best = None
    for w, b1, b2 in product(CNN_WEIGHTS, BIAS_GRID, BIAS_GRID):
        rule = dict(cnn_weight=float(w), bias=[0.0, float(b1), float(b2)])
        f1 = f1_score(yv, decide(cv, sv, rule), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, rule)
    best_val_f1, best_rule = best
    print(f"Chosen on validation: CNN weight = {best_rule['cnn_weight']}, "
          f"bias caries = {best_rule['bias'][1]}, bias deep caries = {best_rule['bias'][2]}  "
          f"(val macro F1 = {best_val_f1:.3f})\n")

    # Ensemble with the same weight but no threshold (to show each step's effect)
    plain_ensemble = dict(cnn_weight=best_rule["cnn_weight"], bias=[0.0, 0.0, 0.0])

    # 3) Evaluate every rule ONCE on the TEST set
    rules = {
        "PCA + SVM (before)": dict(cnn_weight=0.0, bias=[0.0, 0.0, 0.0]),
        "Fine-tuned CNN": dict(cnn_weight=1.0, bias=[0.0, 0.0, 0.0]),
        "Ensemble": plain_ensemble,
        "Ensemble + threshold (final)": best_rule,
    }
    rows = []
    for name, rule in rules.items():
        val_f1 = f1_score(yv, decide(cv, sv, rule), average="macro")
        rows.append(dict(rule=name, val_macro_f1=round(val_f1, 4),
                         **{f"test_{k}": v for k, v in metrics(yt, decide(ct, st, rule)).items()}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "comparison.csv"), index=False)

    pd.set_option("display.width", 250)
    print("=== TEST RESULTS ===")
    print(df.to_string(index=False))
    print("\nFinal rule on the test set:")
    print(classification_report(yt, decide(ct, st, best_rule),
                                target_names=CLASS_NAMES, digits=3))

    # 4) Curve: how a stricter caries threshold trades false alarms for missed caries
    w = best_rule["cnn_weight"]
    p_caries = (w * cv + (1 - w) * sv)[:, 1:].sum(axis=1)
    thresholds = np.linspace(0.05, 0.95, 37)
    prec = [precision_score(yv > 0, p_caries >= t, zero_division=0) for t in thresholds]
    rec = [recall_score(yv > 0, p_caries >= t, zero_division=0) for t in thresholds]
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, prec, label="Precision (fewer false alarms)")
    plt.plot(thresholds, rec, label="Recall (fewer missed caries)")
    plt.xlabel("Caries probability threshold")
    plt.ylabel("Score (validation set)")
    plt.title("Any caries vs. healthy: threshold trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "threshold_curve.png"), dpi=200)
    plt.close()

    # 5) Save the rule for predict.py / app.py
    with open(os.path.join(MODEL_DIR, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(dict(best_rule, val_macro_f1=round(best_val_f1, 4)), f, indent=2)
    print(f"Saved {MODEL_DIR}/decision.json and '{OUT}/' folder.")


if __name__ == "__main__":
    main()