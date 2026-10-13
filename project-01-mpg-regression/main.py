"""
Project 01 - Fuel Efficiency (MPG) Regression
Author : Sanchit Kumar Mishra

Goal
----
Predict a car's fuel efficiency (miles per gallon) from its engine specs
(cylinders, displacement, horsepower, weight) and find out WHICH spec matters.

Data
----
1. Default  : seaborn's 'mpg' dataset (downloaded once by seaborn, then cached).
2. Fallback : a bundled copy of the classic 1974 'mtcars' table
              (data/mtcars_fuel_efficiency.csv) - used automatically when there
              is no internet, or when you pass --offline.

Run
---
    python main.py            # live data if possible, else bundled data
    python main.py --offline  # force the bundled data (fully reproducible)
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (works on any machine)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
FALLBACK_CSV = os.path.join(HERE, "data", "mtcars_fuel_efficiency.csv")

TARGET = "mpg"
FEATURES = ["cylinders", "displacement", "horsepower", "weight"]  # in both data sources


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, human-readable source name).

    The dataframe always has the columns: mpg, cylinders, displacement,
    horsepower, weight (pounds).
    """
    if not offline:
        try:
            import seaborn as sns

            df = sns.load_dataset("mpg")
            df = df[[TARGET] + FEATURES].dropna().reset_index(drop=True)
            return df, f"seaborn 'mpg' ({len(df)} cars, model years 1970-82)"
        except Exception as err:  # no internet, blocked, seaborn missing ...
            print(f"[data] seaborn 'mpg' unavailable ({type(err).__name__}) -> using bundled data")

    df = pd.read_csv(FALLBACK_CSV)[[TARGET] + FEATURES]
    return df, f"bundled mtcars sample ({len(df)} cars, 1974)"


# --------------------------------------------------------------------------- #
# 2. Modelling helpers
# --------------------------------------------------------------------------- #
def cv_scores(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 10) -> dict:
    """Cross-validated R^2 and RMSE.

    On a small dataset (32-400 rows) each test fold is tiny, so scoring every
    fold separately gives wildly noisy R^2.  Instead we collect the out-of-fold
    prediction for EVERY car, score all of them together, and repeat with
    different random shuffles to average out luck.
    """
    r2_list, rmse_list = [], []
    for seed in range(n_repeats):
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        predicted = cross_val_predict(model, X, y, cv=folds)
        r2_list.append(r2_score(y, predicted))
        rmse_list.append(np.sqrt(mean_squared_error(y, predicted)))
    return {"R2": float(np.mean(r2_list)), "RMSE": float(np.mean(rmse_list))}


def gallons_per_100_miles_model() -> TransformedTargetRegressor:
    """Linear model that predicts 100/mpg (gallons per 100 miles), then converts back to mpg.

    100/mpg is 'fuel burned per distance', which is what physics (weight, drag)
    actually acts on - so we expect it to be more linear in weight than mpg is.
    """
    return TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=lambda mpg: 100.0 / mpg,
        inverse_func=lambda gp100: 100.0 / gp100,
        check_inverse=False,
    )


# --------------------------------------------------------------------------- #
# 3. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, source = load_data(offline)
    X, y = df[FEATURES], df[TARGET]
    print(f"[data] source = {source}")
    print(f"[data] shape  = {df.shape}, missing values = {int(df.isna().sum().sum())}\n")

    # --- correlation of each spec with mpg --------------------------------
    corr = df.corr()[TARGET].drop(TARGET).sort_values()
    print("Correlation of each engine spec with mpg:")
    print(corr.round(3).to_string(), "\n")

    # --- compare models with repeated 5-fold cross-validation --------------
    models = {
        "Baseline (always predict mean mpg)": (DummyRegressor(), FEATURES),
        "Linear regression: weight only": (LinearRegression(), ["weight"]),
        "Linear regression: all 4 specs": (LinearRegression(), FEATURES),
        "Ridge (scaled): all 4 specs": (make_pipeline(StandardScaler(), Ridge(alpha=1.0)), FEATURES),
        "Random forest: all 4 specs": (RandomForestRegressor(n_estimators=200, random_state=42), FEATURES),
        "Linear on 100/mpg: weight only": (gallons_per_100_miles_model(), ["weight"]),
    }
    rows = {name: cv_scores(m, df[cols], y) for name, (m, cols) in models.items()}
    results = pd.DataFrame(rows).T.sort_values("R2", ascending=False)
    print("Cross-validated results (5-fold x 10 repeats, pooled out-of-fold predictions):")
    print(results.round(3).to_string(), "\n")

    # --- plots -------------------------------------------------------------
    best_name = results.index[0]
    best_model, best_cols = models[best_name]
    predicted = cross_val_predict(best_model, df[best_cols], y, cv=KFold(5, shuffle=True, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (a) weight vs mpg with straight line and 100/mpg curve
    ax = axes[0]
    ax.scatter(df["weight"], y, alpha=0.7, label="cars")
    grid = pd.DataFrame({"weight": np.linspace(df["weight"].min(), df["weight"].max(), 100)})
    straight = LinearRegression().fit(df[["weight"]], y)
    curved = gallons_per_100_miles_model().fit(df[["weight"]], y)
    ax.plot(grid["weight"], straight.predict(grid), "r--", label="straight line on mpg")
    ax.plot(grid["weight"], curved.predict(grid), "g-", label="straight line on 100/mpg")
    ax.set_xlabel("Weight (lbs)")
    ax.set_ylabel("Miles per gallon")
    ax.set_title("Weight vs fuel efficiency")
    ax.legend()
    # (b) predicted vs actual for the best model
    ax = axes[1]
    ax.scatter(y, predicted, alpha=0.7)
    lims = [min(y.min(), predicted.min()), max(y.max(), predicted.max())]
    ax.plot(lims, lims, "k--", label="perfect prediction")
    ax.set_xlabel("Actual mpg")
    ax.set_ylabel("Predicted mpg (cross-validated)")
    ax.set_title(f"Best model: {best_name}")
    ax.legend()
    fig.suptitle(f"Project 01 - MPG regression | data: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mpg_regression.png"), dpi=120)
    plt.close(fig)

    # --- key insight (computed from the data, so it is always true) --------
    r2_weight = results.loc["Linear regression: weight only", "R2"]
    r2_all = results.loc["Linear regression: all 4 specs", "R2"]
    r2_gp100 = results.loc["Linear on 100/mpg: weight only", "R2"]
    spec_corr = df[FEATURES].corr().loc["weight", ["displacement", "horsepower", "cylinders"]]
    if r2_gp100 > r2_weight:
        insight = (
            f"How you express the target matters: predicting gallons-per-100-miles (100/mpg) instead of mpg "
            f"lifts the weight-only model from R2={r2_weight:.2f} to R2={r2_gp100:.2f}, "
            f"close to the R2={r2_all:.2f} of all 4 specs - and the other specs add little because they are "
            f"strongly correlated with weight (r={spec_corr.min():.2f} to {spec_corr.max():.2f})."
        )
    else:
        insight = (
            f"Weight alone gives R2={r2_weight:.2f} versus R2={r2_all:.2f} for all 4 specs; the extra specs add "
            f"little because they are correlated with weight (r={spec_corr.min():.2f} to {spec_corr.max():.2f})."
        )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'mpg_regression.png')}")
    return {"results": results, "insight": insight, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 01 - MPG regression")
    parser.add_argument("--offline", action="store_true", help="use the bundled dataset only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
