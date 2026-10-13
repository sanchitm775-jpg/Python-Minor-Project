"""
Project 02 - Wine Cultivar Classification
Author : Sanchit Kumar Mishra

Goal
----
Identify which of 3 grape cultivars (class_0 / class_1 / class_2) a wine was made
from, using 13 chemical measurements (alcohol, flavanoids, proline, ...).

Data
----
sklearn.datasets.load_wine - 178 wines from one region in Italy. It ships inside
scikit-learn, so NO download and NO internet are needed.

Run
---
    python main.py
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return features X (DataFrame), labels y (Series) and the class names."""
    bunch = load_wine()
    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="cultivar")
    return X, y, list(bunch.target_names)


def cross_validated_accuracy(model, X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    """Mean and std of accuracy over 5 stratified folds repeated with 5 different shuffles."""
    scores = []
    for seed in range(5):
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores.extend(cross_val_score(model, X, y, cv=folds, scoring="accuracy"))
    return float(np.mean(scores)), float(np.std(scores))


def run(offline: bool = False) -> dict:  # `offline` accepted for a uniform CLI; no network is used here
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, class_names = load_data()
    print(f"[data] sklearn load_wine: {X.shape[0]} wines, {X.shape[1]} features, classes = {y.value_counts().sort_index().to_dict()}\n")

    # ---- 1. compare models (cross-validated accuracy) ---------------------
    models = {
        "kNN (k=5) - raw features": KNeighborsClassifier(n_neighbors=5),
        "kNN (k=5) - scaled features": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Logistic regression (scaled)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM RBF (scaled)": make_pipeline(StandardScaler(), SVC()),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    rows = []
    for name, model in models.items():
        mean_acc, std_acc = cross_validated_accuracy(model, X, y)
        rows.append({"model": name, "accuracy": mean_acc, "std": std_acc})
    results = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
    print("Cross-validated accuracy (5-fold x 5 repeats):")
    print(results.round(3).to_string(), "\n")

    # ---- 2. why does scaling matter?  look at feature ranges --------------
    ranges = (X.max() - X.min()).sort_values(ascending=False)
    biggest, smallest = ranges.index[0], ranges.index[-1]
    variance_share = X.var() / X.var().sum()  # kNN distance is driven by variance
    print(f"Feature ranges: '{biggest}' spans {ranges.iloc[0]:.0f} units, '{smallest}' spans {ranges.iloc[-1]:.2f}.")
    print(f"'{biggest}' alone holds {variance_share[biggest]:.1%} of the total variance, "
          f"so unscaled kNN is basically a '{biggest}'-only classifier.\n")

    # ---- 3. final held-out test with the best model + confusion matrix ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    best_name = results.index[0]
    best_model = models[best_name].fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f"Hold-out test (25% of data) with best model: {best_name}")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    # ---- 4. feature importance from a random forest -----------------------
    forest = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(X, y)
    importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 5 most important features (random forest):")
    print(importance.head(5).round(3).to_string(), "\n")

    # How good is a tiny model that uses only the top 3 features?
    top3 = list(importance.index[:3])
    top3_acc, _ = cross_validated_accuracy(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X[top3], y)
    print(f"Logistic regression using ONLY the top 3 features {top3}: accuracy = {top3_acc:.3f}\n")

    # ---- 5. plots ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, ax=axes[0], colorbar=False)
    axes[0].set_title(f"Confusion matrix (test set)\n{best_name}")
    importance.head(8)[::-1].plot.barh(ax=axes[1], color="tab:purple")
    axes[1].set_title("Random-forest feature importance (top 8)")
    results["accuracy"][::-1].plot.barh(ax=axes[2], color="tab:green", xerr=results["std"][::-1])
    axes[2].set_xlim(0.5, 1.02)
    axes[2].set_title("Cross-validated accuracy by model")
    axes[2].set_ylabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "wine_results.png"), dpi=120)
    plt.close(fig)

    # ---- 6. key insight (computed) ----------------------------------------
    raw_knn = results.loc["kNN (k=5) - raw features", "accuracy"]
    scaled_knn = results.loc["kNN (k=5) - scaled features", "accuracy"]
    insight = (
        f"Feature scaling alone lifts kNN accuracy from {raw_knn:.0%} to {scaled_knn:.0%}: '{biggest}' spans "
        f"{ranges.iloc[0]:.0f} units while '{smallest}' spans {ranges.iloc[-1]:.2f}, so without scaling the distance is "
        f"dominated by one chemical ({variance_share[biggest]:.1%} of total variance). Also, just 3 features {top3} reach {top3_acc:.0%} accuracy."
    )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'wine_results.png')}")
    return {"results": results, "insight": insight, "top3": top3}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 02 - Wine cultivar classification")
    parser.add_argument("--offline", action="store_true", help="(no effect) this project never uses the network")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
