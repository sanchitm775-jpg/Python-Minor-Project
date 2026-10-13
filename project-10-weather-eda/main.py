"""
Project 10 - Systematic EDA of Daily Weather  (uses a FREE API)
Author : Sanchit Kumar Mishra

Goal
----
Run a complete exploratory data analysis (EDA) on five years of daily weather for
Raipur, Chhattisgarh: structure -> missing values -> summary statistics ->
outliers -> correlations -> seasonality, and learn what each step can mislead you about.

Data
----
1. Default  : Open-Meteo Historical Weather API (free, no API key)
              https://archive-api.open-meteo.com/v1/archive   (2019-01-01 .. 2023-12-31)
2. Fallback : a SIMULATED stand-in with the same columns and a Raipur-like monsoon climate
              (fixed random seed).  It only exists so the script runs offline; the console
              and plot titles say "SIMULATED" whenever it is used.

Run
---
    python main.py            # live API if reachable, else simulated stand-in
    python main.py --offline  # force the simulated stand-in
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HEADERS = {"User-Agent": "MastersLabProjects/1.0 (student project by Sanchit Kumar Mishra)"}
LOCATION = {"name": "Raipur, Chhattisgarh", "latitude": 21.25, "longitude": 81.63}
START_DATE, END_DATE = "2019-01-01", "2023-12-31"

# API variable name -> short column name used in this project
API_VARIABLES = {
    "temperature_2m_max": "temp_max_c",
    "temperature_2m_min": "temp_min_c",
    "temperature_2m_mean": "temp_mean_c",
    "precipitation_sum": "rain_mm",
    "wind_speed_10m_max": "wind_max_kmh",
    "shortwave_radiation_sum": "radiation_mj_m2",
    "et0_fao_evapotranspiration": "evaporation_mm",
}
NUMERIC = list(API_VARIABLES.values())
MONSOON_MONTHS = [6, 7, 8, 9]


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Open-Meteo JSON -> DataFrame with a 'date' column and the 7 weather columns.

    JSON null values become NaN automatically, so missing data is preserved (not hidden).
    """
    daily = payload["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_name, column in API_VARIABLES.items():
        df[column] = pd.to_numeric(pd.Series(daily[api_name]), errors="coerce")
    return df


def fetch_open_meteo() -> pd.DataFrame:
    response = requests.get(
        API_URL,
        params={
            "latitude": LOCATION["latitude"], "longitude": LOCATION["longitude"],
            "start_date": START_DATE, "end_date": END_DATE,
            "daily": ",".join(API_VARIABLES), "timezone": "Asia/Kolkata",
        },
        headers=HEADERS, timeout=60,
    )
    response.raise_for_status()
    return parse_open_meteo(response.json())


def simulate_weather(seed: int = 42) -> pd.DataFrame:
    """Fake-but-plausible daily weather with a central-India monsoon pattern (OFFLINE FALLBACK ONLY)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    month = dates.month.to_numpy()
    n = len(dates)

    monthly_mean_temp = np.array([22, 25, 30, 34, 36, 32, 27.5, 27, 28, 27, 24, 21.5])  # typical Raipur shape
    temp_mean = monthly_mean_temp[month - 1] + rng.normal(0, 1.5, n)

    wet_probability = np.array([.03, .04, .04, .04, .06, .45, .80, .82, .60, .15, .03, .02])[month - 1]
    is_wet = rng.random(n) < wet_probability
    rain = np.where(is_wet, rng.gamma(shape=0.6, scale=22.0, size=n), 0.0).round(1)

    temp_max = temp_mean + 5.5 - 0.04 * np.minimum(rain, 60) + rng.normal(0, 0.8, n)
    temp_min = temp_mean - 5.0 + rng.normal(0, 0.8, n)
    radiation = np.clip(24 - 0.18 * np.minimum(rain, 100) - 3 * (month >= 6) * (month <= 9) + rng.normal(0, 2, n), 2, 30)
    wind = np.clip(14 + 6 * (month >= 5) * (month <= 9) + 0.05 * rain + rng.normal(0, 3, n), 3, None)
    evaporation = np.clip(0.2 * radiation + 0.08 * (temp_max - temp_min) + rng.normal(0, 0.4, n), 0.3, None)

    return pd.DataFrame({
        "date": dates, "temp_max_c": temp_max.round(1), "temp_min_c": temp_min.round(1),
        "temp_mean_c": temp_mean.round(1), "rain_mm": rain, "wind_max_kmh": wind.round(1),
        "radiation_mj_m2": radiation.round(2), "evaporation_mm": evaporation.round(2),
    })


def load_data(offline: bool = False) -> tuple[pd.DataFrame, str, bool]:
    if not offline:
        try:
            df = fetch_open_meteo()
            return df, f"Open-Meteo API (live) - {LOCATION['name']}, {START_DATE} to {END_DATE}", False
        except Exception as err:
            print(f"[data] Open-Meteo API unavailable ({type(err).__name__}) -> using SIMULATED stand-in")
    return simulate_weather(), f"SIMULATED stand-in for {LOCATION['name']} (not real data)", True


# --------------------------------------------------------------------------- #
# 2. EDA building blocks
# --------------------------------------------------------------------------- #
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({"missing": df.isna().sum(), "missing_%": (df.isna().mean() * 100).round(2)})
    return report


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count outliers per column with the two textbook rules: IQR (1.5 x IQR) and |z-score| > 3."""
    rows = {}
    for col in NUMERIC:
        values = df[col].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
        z_outliers = ((values - values.mean()).abs() / values.std() > 3).sum()
        rows[col] = {"Q1": q1, "Q3": q3, "IQR_rule": int(iqr_outliers), "IQR_%": round(iqr_outliers / len(values) * 100, 1),
                     "z>3_rule": int(z_outliers), "z>3_%": round(z_outliers / len(values) * 100, 1)}
    return pd.DataFrame(rows).T


def describe(strength: float) -> str:
    """Words for a correlation coefficient."""
    s = abs(strength)
    return "almost no" if s < 0.2 else "a weak" if s < 0.4 else "a moderate" if s < 0.7 else "a strong"


# --------------------------------------------------------------------------- #
# 3. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, source, simulated = load_data(offline)
    banner = "   *** SIMULATED DATA - run with internet for the real API data ***" if simulated else ""
    print(f"[data] source = {source}{banner}\n")

    # ---- Step 1: structure ------------------------------------------------------
    print("STEP 1 - Structure")
    print(f"  rows x columns : {df.shape}")
    print(f"  date range     : {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    expected_days = (df["date"].max() - df["date"].min()).days + 1
    print(f"  days expected  : {expected_days}, days present: {df['date'].nunique()}, duplicate dates: {int(df['date'].duplicated().sum())}")
    print("  column types   :", dict(df.dtypes.astype(str)), "\n")

    # ---- Step 2: missing values ---------------------------------------------------
    print("STEP 2 - Missing values")
    missing = missing_report(df)
    print(missing.to_string())
    print("  -> " + ("no missing values." if missing["missing"].sum() == 0 else
                    f"{int(missing['missing'].sum())} missing cells; they are kept as NaN and skipped in statistics."), "\n")

    # ---- Step 3: summary statistics ---------------------------------------------------
    print("STEP 3 - Summary statistics")
    summary = df[NUMERIC].describe().T
    summary["skew"] = df[NUMERIC].skew()
    print(summary[["mean", "50%", "std", "min", "max", "skew"]].round(2).to_string())
    print("  (a mean far above the median together with a large positive skew means a long right tail)\n")

    # ---- Step 4: outliers ---------------------------------------------------------------
    print("STEP 4 - Outliers (two rules)")
    outliers = outlier_report(df)
    print(outliers.round(2).to_string())
    impossible = int((df["temp_min_c"] > df["temp_max_c"]).sum())
    print(f"  sanity check: days with min temperature above max temperature = {impossible}\n")

    # ---- Step 5: correlations -------------------------------------------------------------
    print("STEP 5 - Correlations")
    pearson, spearman = df[NUMERIC].corr(method="pearson"), df[NUMERIC].corr(method="spearman")
    pairs = [(a, b, pearson.loc[a, b], spearman.loc[a, b]) for i, a in enumerate(NUMERIC) for b in NUMERIC[i + 1:]]
    top = sorted(pairs, key=lambda p: abs(p[2]), reverse=True)[:5]
    print("  strongest Pearson pairs:", "; ".join(f"{a}~{b}: {p:.2f} (Spearman {s:.2f})" for a, b, p, s in top))
    rain_temp_p, rain_temp_s = pearson.loc["rain_mm", "temp_max_c"], spearman.loc["rain_mm", "temp_max_c"]
    print(f"  rain vs max temperature: Pearson {rain_temp_p:.2f}, Spearman {rain_temp_s:.2f}\n")

    # ---- Step 6: seasonality -----------------------------------------------------------------
    print("STEP 6 - Seasonality (monthly climatology)")
    years = df["date"].dt.year.nunique()
    monthly = pd.DataFrame({
        "mean_temp_c": df.groupby(df["date"].dt.month)["temp_mean_c"].mean(),
        "avg_rain_mm_per_year": df.groupby(df["date"].dt.month)["rain_mm"].sum() / years,
    }).round(1)
    monthly.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.index]
    print(monthly.to_string(), "\n")

    # ---- plots ------------------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    if missing["missing"].sum() == 0:
        axes[0, 0].text(0.5, 0.5, "No missing values\nin any column", ha="center", va="center", fontsize=16, color="tab:green")
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
    else:
        missing["missing"].plot.barh(ax=axes[0, 0], color="tab:red")
        axes[0, 0].set_xlabel("number of missing days")
    axes[0, 0].set_title("1. Missing values per column")
    standardized = (df[NUMERIC] - df[NUMERIC].mean()) / df[NUMERIC].std()
    sns.boxplot(data=standardized, orient="h", fliersize=2, ax=axes[0, 1])
    axes[0, 1].set_title("2. Outliers: boxplots of z-scores")
    axes[0, 1].set_xlabel("standard deviations from the mean")
    wet = df.loc[df["rain_mm"] > 0, "rain_mm"]
    axes[0, 2].hist(wet, bins=np.logspace(np.log10(max(wet.min(), 0.1)), np.log10(max(wet.max(), 1)), 40), color="tab:blue")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_title(f"3. Rain on wet days (log x-axis)\n{(df['rain_mm'] == 0).mean():.0%} of days have no rain at all")
    axes[0, 2].set_xlabel("rain (mm)")
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 0], cbar=False, annot_kws={"size": 7})
    axes[1, 0].set_title("4. Pearson correlation (linear)")
    sns.heatmap(spearman, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1, 1], cbar=False, annot_kws={"size": 7})
    axes[1, 1].set_title("5. Spearman correlation (rank-based)")
    ax = axes[1, 2]
    ax.bar(monthly.index, monthly["avg_rain_mm_per_year"], color="tab:blue", alpha=0.7)
    ax.set_ylabel("average rain per year (mm)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(monthly.index, monthly["mean_temp_c"], "o-", color="tab:red")
    ax2.set_ylabel("mean temperature (C)", color="tab:red")
    ax2.grid(False)
    ax.set_title("6. Seasonality: rain bars vs temperature line")
    fig.suptitle(f"Project 10 - weather EDA | data: {source}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT_DIR, "weather_eda.png"), dpi=110)
    plt.close(fig)
    df.to_csv(os.path.join(OUT_DIR, "weather_clean.csv"), index=False)

    # ---- key insight (computed) -------------------------------------------------------------------------
    rain = df["rain_mm"].fillna(0)
    top_decile_days = max(int(len(rain) * 0.10), 1)
    top10_share = rain.sort_values(ascending=False).head(top_decile_days).sum() / rain.sum() * 100
    monsoon_share = rain[df["date"].dt.month.isin(MONSOON_MONTHS)].sum() / rain.sum() * 100
    rain_q3, rain_iqr_pct = outliers.loc["rain_mm", "Q3"], outliers.loc["rain_mm", "IQR_%"]
    hottest = monthly["mean_temp_c"].idxmax()
    wettest = monthly["avg_rain_mm_per_year"].idxmax()
    # are the IQR "outliers" in rain concentrated in the monsoon months?  (evidence for real weather vs errors)
    rain_q1 = rain.quantile(0.25)
    rain_flagged = rain > rain_q3 + 1.5 * (rain_q3 - rain_q1)
    flagged_in_monsoon = df.loc[rain_flagged, "date"].dt.month.isin(MONSOON_MONTHS).mean() * 100 if rain_flagged.any() else 0.0
    verdict = ("so they are genuine monsoon weather, not errors - deleting them would erase the monsoon"
               if flagged_in_monsoon >= 60 else "so they should be inspected before anything is deleted")
    near_zero = (
        f" (Q3 = {rain_q3:.1f} mm, so almost every rainy day looks extreme)" if rain_q3 < 1.0 else ""
    )
    iqr_sentence = (
        f"The textbook IQR rule flags {rain_iqr_pct:.0f}% of days as rain 'outliers'{near_zero}, yet "
        f"{flagged_in_monsoon:.0f}% of them fall in June-September, {verdict}"
    )
    insight = (
        f"Rain is concentrated: the wettest 10% of days deliver {top10_share:.0f}% of all rain and June-September alone "
        f"{monsoon_share:.0f}%. {iqr_sentence}. The hottest month ({hottest}) is not the wettest ({wettest}), so max "
        f"temperature and rain share {describe(rain_temp_p)} linear relationship (Pearson {rain_temp_p:.2f}, Spearman {rain_temp_s:.2f})."
    )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'weather_eda.png')}")
    return {"insight": insight, "source": source, "simulated": simulated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 10 - weather EDA (Open-Meteo API)")
    parser.add_argument("--offline", action="store_true", help="use the simulated stand-in only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
