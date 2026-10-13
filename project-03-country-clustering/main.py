"""
Project 03 - Segmenting Countries by Socio-economic Indicators  (uses a FREE API)
Author : Sanchit Kumar Mishra

Goal
----
Group countries into "development tiers" with K-Means using four indicators:
GDP per capita, life expectancy, adult literacy and infant mortality.

Data
----
1. Default  : World Bank Open Data API  https://api.worldbank.org/v2   (free, no API key)
2. Fallback : data/countries_fallback.csv - a small, ROUNDED, APPROXIMATE snapshot
              (~2021) of 65 countries.  Used when offline or with --offline.
              It is only a safety net so the script always runs; use the live API
              for exact numbers.

Run
---
    python main.py            # live API if reachable, else fallback
    python main.py --offline  # force the fallback snapshot
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
FALLBACK_CSV = os.path.join(HERE, "data", "countries_fallback.csv")

WB_URL = "https://api.worldbank.org/v2"
HEADERS = {"User-Agent": "MastersLabProjects/1.0 (student project by Sanchit Kumar Mishra)"}
# World Bank indicator code -> our column name, and the years to search for the latest value
INDICATORS = {
    "NY.GDP.PCAP.CD": ("gdp_per_capita", "2015:2023"),   # GDP per capita, current US$
    "SP.DYN.LE00.IN": ("life_expectancy", "2015:2023"),  # life expectancy at birth, years
    "SE.ADT.LITR.ZS": ("adult_literacy", "2005:2023"),   # adult literacy rate, % (sparse!)
    "SP.DYN.IMRT.IN": ("infant_mortality", "2015:2023"), # infant deaths per 1,000 births
}
FEATURES = ["log_gdp", "life_expectancy", "adult_literacy", "infant_mortality"]


# --------------------------------------------------------------------------- #
# 1. Data loading (API first, fallback second)
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict) -> list:
    """GET a World Bank endpoint and return the decoded JSON ([metadata, rows])."""
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_country_list(payload: list) -> pd.DataFrame:
    """World Bank /country payload -> DataFrame(iso3, country, region, income_level).

    The API also returns aggregates such as 'World' or 'Euro area'; those have
    region id 'NA' and are dropped, so only real countries remain.
    """
    rows = []
    for item in payload[1]:
        if item["region"]["id"] == "NA":  # aggregate, not a country
            continue
        rows.append({
            "iso3": item["id"],
            "country": item["name"],
            "region": item["region"]["value"].strip(),
            "income_level": item["incomeLevel"]["value"],
        })
    return pd.DataFrame(rows)


def parse_indicator(payload: list, column: str) -> pd.Series:
    """World Bank indicator payload -> Series(index=iso3, values=latest non-null value)."""
    rows = payload[1] or []
    records = [
        {"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
        for r in rows
        if r.get("value") is not None and r.get("countryiso3code")
    ]
    if not records:
        return pd.Series(dtype=float, name=column)
    frame = pd.DataFrame(records).sort_values("year")
    return frame.groupby("iso3")["value"].last().rename(column)  # last = most recent year


def fetch_world_bank() -> pd.DataFrame:
    """Download the 4 indicators for all countries and merge them into one table."""
    countries = parse_country_list(_get_json(f"{WB_URL}/country", {"format": "json", "per_page": 400}))
    table = countries.set_index("iso3")
    for code, (column, years) in INDICATORS.items():
        payload = _get_json(
            f"{WB_URL}/country/all/indicator/{code}",
            {"format": "json", "per_page": 20000, "date": years},
        )
        table = table.join(parse_indicator(payload, column))
    # Rich countries often have no literacy figure (it is assumed ~universal), so treat
    # 'High income' countries with a missing value as 99 %. Documented assumption!
    rich = table["income_level"].eq("High income") & table["adult_literacy"].isna()
    table.loc[rich, "adult_literacy"] = 99.0
    return table.reset_index(drop=True)


def load_data(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (clean dataframe, source description)."""
    if not offline:
        try:
            raw = fetch_world_bank()
            return raw, "World Bank API (live)"
        except Exception as err:
            print(f"[data] World Bank API unavailable ({type(err).__name__}) -> using bundled snapshot")
    raw = pd.read_csv(FALLBACK_CSV)
    return raw, "bundled APPROXIMATE snapshot (~2021, 65 countries)"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep complete rows, add log10(GDP) because income is extremely right-skewed."""
    needed = ["gdp_per_capita", "life_expectancy", "adult_literacy", "infant_mortality"]
    df = raw.dropna(subset=needed).copy()
    df = df[df["gdp_per_capita"] > 0]
    df["log_gdp"] = np.log10(df["gdp_per_capita"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Clustering
# --------------------------------------------------------------------------- #
def choose_k(X_scaled: np.ndarray, k_values=range(3, 8)) -> tuple[int, dict]:
    """Pick k (3..7) with the best silhouette score. We start at 3 so that we get
    more than just a 'rich vs poor' split."""
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return max(scores, key=scores.get), scores


def name_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters by average life expectancy and give them readable names."""
    order = df.groupby("cluster")["life_expectancy"].mean().sort_values().index
    names = {cluster: f"Tier {rank + 1}" for rank, cluster in enumerate(order)}  # Tier 1 = lowest
    df["tier"] = df["cluster"].map(names)
    return df


# --------------------------------------------------------------------------- #
# 3. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw, source = load_data(offline)
    df = clean(raw)
    print(f"[data] source = {source}")
    print(f"[data] {len(raw)} countries downloaded/loaded, {len(df)} with all 4 indicators -> used for clustering\n")

    X_scaled = StandardScaler().fit_transform(df[FEATURES])
    best_k, sil_scores = choose_k(X_scaled)
    print("Silhouette score by k:", {k: round(v, 3) for k, v in sil_scores.items()}, f"-> best k = {best_k}\n")

    df["cluster"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_scaled)
    df = name_clusters(df)

    # How different is the 4-indicator clustering from a GDP-only clustering? (1.0 = identical)
    gdp_only = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_scaled[:, [FEATURES.index("log_gdp")]])
    agreement = adjusted_rand_score(df["cluster"], gdp_only)

    profile = df.groupby("tier").agg(
        countries=("country", "count"),
        gdp_per_capita=("gdp_per_capita", "median"),
        life_expectancy=("life_expectancy", "mean"),
        adult_literacy=("adult_literacy", "mean"),
        infant_mortality=("infant_mortality", "mean"),
    ).round(1)
    print("Cluster profile (GDP = median, others = mean):")
    print(profile.to_string(), "\n")
    for tier, group in df.groupby("tier"):
        sample = ", ".join(group.sort_values("gdp_per_capita")["country"].head(6))
        print(f"  {tier}: {sample}{' ...' if len(group) > 6 else ''}")
    print()

    # --- "Income vs health" analysis: the basis of our key insight ---------
    slope, intercept = np.polyfit(df["log_gdp"], df["life_expectancy"], 1)
    df["expected_life"] = slope * df["log_gdp"] + intercept
    df["life_gap"] = df["life_expectancy"] - df["expected_life"]  # + = healthier than income predicts
    over = df.sort_values("life_gap", ascending=False).head(3)
    under = df.sort_values("life_gap").head(3)
    cut = df["gdp_per_capita"].median()
    low, high = df[df["gdp_per_capita"] <= cut], df[df["gdp_per_capita"] > cut]
    r_low = low["log_gdp"].corr(low["life_expectancy"])
    r_high = high["log_gdp"].corr(high["life_expectancy"])
    print(f"Correlation (log GDP vs life expectancy): poorer half = {r_low:.2f}, richer half = {r_high:.2f}")
    print("Healthier than their income predicts:", ", ".join(f"{r.country} (+{r.life_gap:.1f} yrs)" for r in over.itertuples()))
    print("Less healthy than their income predicts:", ", ".join(f"{r.country} ({r.life_gap:.1f} yrs)" for r in under.itertuples()), "\n")

    # --- save table + plots ------------------------------------------------
    df.drop(columns=["cluster"]).to_csv(os.path.join(OUT_DIR, "country_clusters.csv"), index=False)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    for tier, group in df.groupby("tier"):
        ax.scatter(coords[group.index, 0], coords[group.index, 1], label=tier, s=45, alpha=0.8)
    for idx in df.sort_values("gdp_per_capita").index[[0, -1]].tolist() + over.index.tolist():
        ax.annotate(df.loc[idx, "country"], coords[idx], fontsize=8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.set_title(f"Countries grouped into {best_k} tiers (PCA view)")
    ax.legend()
    ax = axes[1]
    for tier, group in df.groupby("tier"):
        ax.scatter(group["gdp_per_capita"], group["life_expectancy"], label=tier, s=45, alpha=0.8)
    xs = np.linspace(df["log_gdp"].min(), df["log_gdp"].max(), 100)
    ax.plot(10 ** xs, slope * xs + intercept, "k--", label="fitted income-health curve")
    ax.set_xscale("log")
    ax.set_xlabel("GDP per capita (US$, log scale)")
    ax.set_ylabel("Life expectancy (years)")
    ax.set_title("Income vs life expectancy")
    ax.legend()
    fig.suptitle(f"Project 03 - country clustering | data: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "country_clusters.png"), dpi=120)
    plt.close(fig)

    # --- key insight (computed) --------------------------------------------
    r_squared = df["log_gdp"].corr(df["life_expectancy"]) ** 2
    insight = (
        f"Income explains only part of health: log-GDP alone accounts for R2={r_squared:.2f} of the variation in life "
        f"expectancy, leaving gaps of up to {df['life_gap'].abs().max():.0f} years. "
        f"{', '.join(over['country'])} live {over['life_gap'].min():.1f}-{over['life_gap'].max():.1f} years LONGER than their "
        f"income predicts, while {', '.join(under['country'])} live {abs(under['life_gap'].max()):.1f}-"
        f"{abs(under['life_gap'].min()):.1f} years SHORTER. Clusters built on all 4 indicators agree with a GDP-only "
        f"clustering only partly (adjusted Rand index = {agreement:.2f}, where 1.0 means identical)."
    )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'country_clusters.png')}")
    return {"profile": profile, "insight": insight, "best_k": best_k, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 03 - country clustering (World Bank API)")
    parser.add_argument("--offline", action="store_true", help="use the bundled snapshot only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
