"""
run_experiments.py

Stage 2: feature selection + caries classification (healthy / caries / deep caries).

Compares 5 feature selection settings x 3 classifiers:
  Feature selection: None (all 2048 features), PCA, ANOVA F-test,
                     Mutual Information, RFE (logistic regression)
  Classifiers:       Logistic Regression, SVM (RBF), Random Forest
All classifiers use balanced class weights (healthy teeth are the majority).

For every (method, classifier) pair the number of features k is chosen on the
validation set (macro F1). The chosen setting is then evaluated ONCE on the
test set. Feature rankings are computed on the training set only.

Run after extract_features.py.
Output: results/results.csv          -> test results of every pair
        results/val_curves.csv       -> validation macro F1 for every k
        results/best_model.joblib    -> best pipeline (chosen by validation F1)
        results/confusion_matrix.png, results/f1_vs_k.png, results/report.txt
"""
import os
import time
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

FEAT = "features"
OUT = "results"
SEED = 42
K_VALUES = [64, 128, 256, 512]
METHODS = ["None", "PCA", "ANOVA", "MutualInfo", "RFE"]
CLASS_NAMES = ["Healthy", "Caries", "Deep caries"]

CLASSIFIERS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM": lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
}


def load(split):
    return np.load(f"{FEAT}/{split}_X.npy"), np.load(f"{FEAT}/{split}_y.npy")


def main():
    os.makedirs(OUT, exist_ok=True)
    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    X_te, y_te = load("test")

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(x) for x in (X_tr, X_va, X_te))

    # 1) Feature rankings, computed once on the training set
    print("Ranking features (this takes a few minutes)...")
    t = time.time()
    ranking = {
        "ANOVA": np.argsort(-np.nan_to_num(f_classif(X_tr, y_tr)[0])),
        "MutualInfo": np.argsort(-mutual_info_classif(X_tr, y_tr, random_state=SEED)),
    }
    rfe = RFE(LogisticRegression(max_iter=500, class_weight="balanced"),
              n_features_to_select=min(K_VALUES), step=0.2).fit(X_tr, y_tr)
    ranking["RFE"] = np.argsort(rfe.ranking_, kind="stable")
    pca = PCA(n_components=max(K_VALUES), random_state=SEED).fit(X_tr)
    print(f"  done in {time.time() - t:.0f}s\n")

    def select(method, k):
        """Returns train/val/test matrices with k features and a selector description."""
        if method == "None":
            return X_tr, X_va, X_te, ("none", None)
        if method == "PCA":
            p = [pca.transform(x)[:, :k] for x in (X_tr, X_va, X_te)]
            return (*p, ("pca", k))
        idx = ranking[method][:k]
        return X_tr[:, idx], X_va[:, idx], X_te[:, idx], ("idx", idx)

    # 2) Train / validate / test every combination
    results, curves, best_overall = [], [], None
    for method in METHODS:
        ks = [X_tr.shape[1]] if method == "None" else K_VALUES
        for clf_name, make_clf in CLASSIFIERS.items():
            best = None
            for k in ks:
                a, b, c, selector = select(method, k)
                t = time.time()
                clf = make_clf().fit(a, y_tr)
                val_f1 = f1_score(y_va, clf.predict(b), average="macro")
                curves.append(dict(method=method, classifier=clf_name, k=k, val_macro_f1=val_f1))
                print(f"  {method:10s} {clf_name:12s} k={k:4d}  val macro-F1={val_f1:.3f}"
                      f"  ({time.time() - t:.0f}s)")
                if best is None or val_f1 > best["val_f1"]:
                    best = dict(k=k, val_f1=val_f1, clf=clf, selector=selector, X_test=c)

            pred = best["clf"].predict(best["X_test"])
            per_class = f1_score(y_te, pred, average=None, labels=[0, 1, 2])
            results.append(dict(
                method=method, classifier=clf_name, k=best["k"],
                val_macro_f1=round(best["val_f1"], 4),
                test_accuracy=round(accuracy_score(y_te, pred), 4),
                test_balanced_accuracy=round(balanced_accuracy_score(y_te, pred), 4),
                test_macro_f1=round(f1_score(y_te, pred, average="macro"), 4),
                f1_healthy=round(per_class[0], 4),
                f1_caries=round(per_class[1], 4),
                f1_deep_caries=round(per_class[2], 4),
            ))
            if best_overall is None or best["val_f1"] > best_overall["val_f1"]:
                best_overall = dict(best, method=method, classifier=clf_name, pred=pred)

    # 3) Save tables
    df = pd.DataFrame(results)
    df.to_csv(f"{OUT}/results.csv", index=False)
    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(f"{OUT}/val_curves.csv", index=False)

    b = best_overall
    report = classification_report(y_te, b["pred"], target_names=CLASS_NAMES, digits=3)
    with open(f"{OUT}/report.txt", "w") as f:
        f.write(f"Best (chosen on validation): {b['method']} + {b['classifier']}, k={b['k']}\n\n")
        f.write(report)

    # 4) Save the best pipeline for predict.py
    joblib.dump(dict(
        scaler=scaler,
        method=b["method"],
        selector=b["selector"],
        pca=pca if b["method"] == "PCA" else None,
        classifier=b["clf"],
        class_names=CLASS_NAMES,
    ), f"{OUT}/best_model.joblib")

    # 5) Figures
    ConfusionMatrixDisplay.from_predictions(y_te, b["pred"], display_labels=CLASS_NAMES,
                                            cmap="Blues", colorbar=False)
    plt.title(f"Test set: {b['method']} + {b['classifier']} (k={b['k']})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrix.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in METHODS[1:]:
        d = curves_df[(curves_df.method == method) & (curves_df.classifier == b["classifier"])]
        ax.plot(d.k, d.val_macro_f1, marker="o", label=method)
    base = curves_df[(curves_df.method == "None") & (curves_df.classifier == b["classifier"])]
    ax.axhline(base.val_macro_f1.iloc[0], ls="--", color="gray", label="All 2048 features")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.set_xlabel("Number of selected features (k)")
    ax.set_ylabel("Validation macro F1")
    ax.set_title(f"Feature selection comparison ({b['classifier']})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/f1_vs_k.png", dpi=200)
    plt.close()

    # 6) Summary
    pd.set_option("display.width", 200)
    print("\n=== TEST RESULTS (k chosen on validation) ===")
    print(df.sort_values("val_macro_f1", ascending=False).to_string(index=False))
    print(f"\nBest model: {b['method']} + {b['classifier']} (k={b['k']})\n")
    print(report)
    print(f"Saved to '{OUT}/'")


if __name__ == "__main__":
    main()