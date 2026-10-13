"""
Project 06 - Airline Passenger Forecasting  (time series)
Author : Sanchit Kumar Mishra

Goal
----
Forecast monthly airline passengers with simple, explainable methods:
moving-average trend, linear trend + seasonality, and exponential trend + seasonality.
Find out which assumption about seasonality is right.

Data
----
Monthly international airline passengers (thousands), Jan 1949 - Dec 1960
(144 values) - the classic "AirPassengers" series, identical to
seaborn.load_dataset('flights').  It is bundled in data/flights.csv, so no
download and no internet are needed.

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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
DATA_CSV = os.path.join(HERE, "data", "flights.csv")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_series() -> pd.Series:
    """Read the CSV and return a monthly pandas Series indexed by the first day of each month."""
    df = pd.read_csv(DATA_CSV)
    df["month_number"] = df["month"].map({name: i + 1 for i, name in enumerate(MONTHS)})
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month_number"], day=1))
    series = df.set_index("date")["passengers"].astype(float).sort_index()
    series.index.freq = "MS"
    return series


# --------------------------------------------------------------------------- #
# 2. Forecasting methods (all written with plain numpy so every step is visible)
# --------------------------------------------------------------------------- #
def seasonal_naive(train: pd.Series, future_index: pd.DatetimeIndex) -> pd.Series:
    """Forecast = the value from the same month one year earlier."""
    last_year = train.iloc[-12:].to_numpy()
    values = [last_year[(d.month - train.index[-12].month) % 12] for d in future_index]
    return pd.Series(values, index=future_index)


def trend_plus_seasonality(train: pd.Series, future_index: pd.DatetimeIndex, multiplicative: bool) -> pd.Series:
    """Fit a trend line, then add (or multiply by) an average seasonal pattern.

    additive       : y = a + b*t + season[month]                (straight-line trend)
    multiplicative : log(y) = a + b*t + season[month]  -> y = exp(a+b*t) * seasonal factor
                     i.e. an exponential trend whose seasonal swings grow with the level.
    """
    t_train = np.arange(len(train))
    t_future = np.arange(len(train), len(train) + len(future_index))
    y = np.log(train.to_numpy()) if multiplicative else train.to_numpy()

    slope, intercept = np.polyfit(t_train, y, 1)
    residual = y - (intercept + slope * t_train)
    season = pd.Series(residual, index=train.index.month).groupby(level=0).mean()  # average residual per calendar month

    prediction = intercept + slope * t_future + season.reindex(future_index.month).to_numpy()
    if multiplicative:
        prediction = np.exp(prediction)
    return pd.Series(prediction, index=future_index)


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    """MAE (thousand passengers), RMSE and MAPE (% error) on the test period."""
    error = actual - predicted
    return {
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt((error ** 2).mean())),
        "MAPE_%": float((error.abs() / actual).mean() * 100),
    }


def centered_moving_average(series: pd.Series) -> pd.Series:
    """2x12 centred moving average: smooths out one full seasonal cycle, leaving the trend."""
    return series.rolling(12).mean().rolling(2).mean().shift(-6)


METHODS = {
    "Seasonal naive (repeat last year)": lambda train, idx: seasonal_naive(train, idx),
    "Linear trend + additive seasonality": lambda train, idx: trend_plus_seasonality(train, idx, multiplicative=False),
    "Exponential trend + multiplicative seasonality": lambda train, idx: trend_plus_seasonality(train, idx, multiplicative=True),
}


def rolling_backtest(series: pd.Series, n_folds: int = 5) -> tuple[pd.DataFrame, dict]:
    """Time-series cross-validation ("rolling origin").

    For each of the last `n_folds` years: train ONLY on the data before that year,
    forecast the 12 months of that year, and score it.  One test year can be a fluke;
    five different years give a fairer comparison.

    Returns (one row per year x method, forecasts of the most recent year for plotting).
    """
    rows, last_year_forecasts = [], {}
    for k in range(n_folds, 0, -1):
        train = series.iloc[: len(series) - 12 * k]
        test = series.iloc[len(series) - 12 * k: len(series) - 12 * (k - 1)]
        for name, method in METHODS.items():
            prediction = method(train, test.index)
            rows.append({"year": test.index[0].year, "method": name, **score(test, prediction)})
            if k == 1:
                last_year_forecasts[name] = prediction
    return pd.DataFrame(rows), last_year_forecasts


# --------------------------------------------------------------------------- #
# 3. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:  # `offline` accepted for a uniform CLI; this project is always offline
    os.makedirs(OUT_DIR, exist_ok=True)
    series = load_series()
    print(f"[data] {len(series)} monthly values, {series.index[0]:%b %Y} to {series.index[-1]:%b %Y}, "
          f"min={series.min():.0f}, max={series.max():.0f}\n")

    # ---- trend via moving average --------------------------------------------
    trend_ma = centered_moving_average(series)

    # ---- rolling-origin backtest: forecast each of the last 5 years from earlier data ----
    test = series.iloc[-12:]
    folds, forecasts = rolling_backtest(series, n_folds=5)
    per_year = folds.pivot(index="year", columns="method", values="MAPE_%")
    table = folds.groupby("method")[["MAE", "RMSE", "MAPE_%"]].mean().sort_values("MAPE_%")
    print("MAPE (%) for each test year (model trained only on earlier years):")
    print(per_year.round(1).to_string(), "\n")
    print("Average over the 5 test years:")
    print(table.round(2).to_string(), "\n")
    best_name = table.index[0]

    # ---- how big are the seasonal swings each year? ---------------------------
    yearly = series.groupby(series.index.year).agg(["mean", "min", "max"])
    yearly["swing"] = yearly["max"] - yearly["min"]
    yearly["swing_%_of_mean"] = yearly["swing"] / yearly["mean"] * 100
    print("Seasonal swing (peak month minus lowest month) per year:")
    print(yearly[["mean", "swing", "swing_%_of_mean"]].round(1).to_string(), "\n")

    # ---- final forecast: refit the winning approach on ALL data, predict 1961 ----
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    final_forecast = trend_plus_seasonality(series, future_index, multiplicative=("multiplicative" in best_name))
    print(f"Forecast for {future_index[0]:%Y} using '{best_name}' (thousand passengers):")
    print(final_forecast.round(0).astype(int).to_string(), "\n")

    # ---- plots -----------------------------------------------------------------
    t = np.arange(len(series))
    lin_slope, lin_intercept = np.polyfit(t, series.to_numpy(), 1)
    exp_slope, exp_intercept = np.polyfit(t, np.log(series.to_numpy()), 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    ax = axes[0, 0]
    ax.plot(series.index, series, label="monthly passengers", color="tab:blue")
    ax.plot(trend_ma.index, trend_ma, label="12-month moving average (trend)", color="red", lw=2)
    ax.plot(series.index, lin_intercept + lin_slope * t, "g--", label="straight-line trend")
    ax.plot(series.index, np.exp(exp_intercept + exp_slope * t), "m--", label="exponential trend")
    ax.set_title("1. Series, moving average and fitted trends")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(test.index, test, "ko-", label="actual 1960", lw=2)
    for name, pred in forecasts.items():
        ax.plot(pred.index, pred, "o--", label=f"{name} (1960 MAPE {per_year.loc[test.index[0].year, name]:.1f}%)")
    ax.set_title("2. Forecasting 1960 from data up to 1959")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(series.index, series, color="tab:blue", label="history")
    ax.plot(final_forecast.index, final_forecast, "r-o", label=f"forecast ({best_name})")
    ax.axvline(series.index[-1], color="gray", ls=":")
    ax.set_title(f"3. Forecast for {future_index[0]:%Y}")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(yearly.index, yearly["swing"], color="tab:orange", label="seasonal swing (peak - low)")
    ax.set_ylabel("Swing in passengers (thousands)")
    ax2 = ax.twinx()
    ax2.plot(yearly.index, yearly["swing_%_of_mean"], "ko-", label="swing as % of yearly average")
    ax2.set_ylabel("Swing as % of yearly mean")
    ax2.set_ylim(0, max(80, yearly["swing_%_of_mean"].max() * 1.2))
    ax2.grid(False)
    ax.set_title("4. Seasonal swings grow with the level")
    fig.suptitle("Project 06 - airline passenger forecasting (1949-1960)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "flights_forecast.png"), dpi=120)
    plt.close(fig)

    # ---- key insight (computed) ---------------------------------------------------
    first, last = yearly.index[0], yearly.index[-1]
    additive_mape = table.loc["Linear trend + additive seasonality", "MAPE_%"]
    multiplicative_mape = table.loc["Exponential trend + multiplicative seasonality", "MAPE_%"]
    growth = f"The peak-to-low seasonal swing grew from {yearly.loc[first, 'swing']:.0f}k in {first} to " \
             f"{yearly.loc[last, 'swing']:.0f}k in {last} ({yearly.loc[last, 'swing'] / yearly.loc[first, 'swing']:.1f}x), " \
             f"but stayed around {yearly['swing_%_of_mean'].mean():.0f}% of each year's average - the seasonality scales with " \
             f"the level (multiplicative), so a fixed-size additive pattern cannot fit every year. "
    mult_name, add_name = "Exponential trend + multiplicative seasonality", "Linear trend + additive seasonality"
    mult_wins = [int(y) for y in per_year.index if per_year.loc[y, mult_name] < per_year.loc[y, add_name]]
    add_wins = [int(y) for y in per_year.index if y not in mult_wins]
    avg_winner = "multiplicative" if multiplicative_mape < additive_mape else "additive"
    verdict = (
        f"On average over the 5 test years the {avg_winner} model has the lower error (MAPE {multiplicative_mape:.1f}% "
        f"multiplicative vs {additive_mape:.1f}% additive), but the winner flips from year to year: multiplicative is better in "
        f"{mult_wins or 'no year'}, additive in {add_wins or 'no year'} - a single-year backtest could have crowned either model."
    )
    insight = growth + verdict
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'flights_forecast.png')}")
    return {"table": table, "insight": insight}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 06 - flights forecasting")
    parser.add_argument("--offline", action="store_true", help="(no effect) this project always uses its bundled CSV")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
