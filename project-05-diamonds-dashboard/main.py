"""
Project 05 - Diamonds Price Dashboard  (data visualization)
Author : Sanchit Kumar Mishra

Goal
----
Build one six-panel dashboard that explains what drives diamond prices
(carat, cut, colour, clarity) and fix a classic misreading of the data.

Data
----
1. Default  : seaborn's 'diamonds' dataset (53,940 diamonds; downloaded once by seaborn).
2. Fallback : a SIMULATED stand-in with the same columns, generated with a fixed seed.
              It is only there so the script runs without internet (e.g. in a lab);
              the console and the chart title say "SIMULATED" whenever it is used.

Run
---
    python main.py            # real data if possible, else simulated stand-in
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
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

# Quality orders: first = worst, last = best
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]          # J = most yellow, D = colourless
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
NUMERIC_COLUMNS = ["carat", "depth", "table", "price", "x", "y", "z"]


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def simulate_diamonds(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Make a fake-but-realistic diamonds table (used ONLY when the real data is unreachable).

    Built-in relationships: price grows ~carat^1.7; better colour/clarity cost more;
    big stones are less likely to be cut 'Ideal' (as in the real data).
    """
    rng = np.random.default_rng(seed)
    carat = np.clip(rng.lognormal(mean=np.log(0.7), sigma=0.55, size=n), 0.23, 3.5).round(2)

    base_cut_weights = np.array([0.03, 0.09, 0.22, 0.26, 0.40])
    cut = []
    for c in carat:
        # larger stones: shift probability away from 'Ideal' towards 'Fair/Good/Premium'
        shift = np.exp(np.array([0.9, 0.5, 0.0, 0.2, -1.1]) * (c - 0.7))
        weights = base_cut_weights * shift
        cut.append(rng.choice(CUT_ORDER, p=weights / weights.sum()))
    color = rng.choice(COLOR_ORDER, size=n, p=[0.06, 0.11, 0.17, 0.23, 0.18, 0.16, 0.09])
    clarity = rng.choice(CLARITY_ORDER, size=n, p=[0.01, 0.17, 0.24, 0.23, 0.15, 0.09, 0.07, 0.04])

    cut_mult = dict(zip(CUT_ORDER, [0.90, 0.95, 1.00, 1.03, 1.08]))
    color_mult = dict(zip(COLOR_ORDER, [0.80, 0.88, 0.95, 1.00, 1.08, 1.15, 1.25]))
    clarity_mult = dict(zip(CLARITY_ORDER, [0.60, 0.80, 0.90, 1.00, 1.10, 1.22, 1.32, 1.45]))
    price = (
        3800 * carat ** 1.7
        * np.array([cut_mult[v] for v in cut])
        * np.array([color_mult[v] for v in color])
        * np.array([clarity_mult[v] for v in clarity])
        * rng.lognormal(0, 0.12, n)
    ).round()
    price = np.maximum(price, 326 + rng.integers(0, 60, n))  # real data starts at $326; avoid a hard pile-up

    depth = rng.normal(61.7, 1.4, n).round(1)
    x = (6.45 * carat ** (1 / 3) * rng.normal(1, 0.01, n)).round(2)
    return pd.DataFrame({
        "carat": carat, "cut": cut, "color": color, "clarity": clarity, "depth": depth,
        "table": rng.normal(57.5, 2.2, n).round(1), "price": price,
        "x": x, "y": (x * rng.normal(1, 0.005, n)).round(2), "z": (x * depth / 100).round(2),
    })


def load_data(offline: bool = False) -> tuple[pd.DataFrame, str, bool]:
    """Return (dataframe, source description, is_simulated)."""
    if not offline:
        try:
            df = sns.load_dataset("diamonds")
            return df, f"seaborn 'diamonds' ({len(df):,} real diamonds)", False
        except Exception as err:
            print(f"[data] seaborn 'diamonds' unavailable ({type(err).__name__}) -> using SIMULATED stand-in")
    df = simulate_diamonds()
    return df, f"SIMULATED stand-in ({len(df):,} fake diamonds - not real data)", True


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible rows and turn quality columns into ordered categories."""
    df = df.copy()
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]  # real data has a few 0-size rows
    df["cut"] = pd.Categorical(df["cut"].astype(str), categories=CUT_ORDER, ordered=True)
    df["color"] = pd.Categorical(df["color"].astype(str), categories=COLOR_ORDER, ordered=True)
    df["clarity"] = pd.Categorical(df["clarity"].astype(str), categories=CLARITY_ORDER, ordered=True)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Dashboard
# --------------------------------------------------------------------------- #
def build_dashboard(df: pd.DataFrame, source: str, path: str) -> None:
    """Draw 6 panels into one PNG."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    # (1) price distribution - heavily skewed, so use a log x-axis
    ax = axes[0, 0]
    ax.hist(df["price"], bins=np.logspace(np.log10(df["price"].min()), np.log10(df["price"].max()), 50), color="tab:blue")
    ax.set_xscale("log")
    ax.axvline(df["price"].median(), color="red", ls="--", label=f"median ${df['price'].median():,.0f}")
    ax.axvline(df["price"].mean(), color="orange", ls="--", label=f"mean ${df['price'].mean():,.0f}")
    ax.set_title("1. Price distribution (log scale)")
    ax.set_xlabel("Price (US$)")
    ax.set_ylabel("Number of diamonds")
    ax.legend()

    # (2) carat vs price - density hexbin on log axes
    ax = axes[0, 1]
    hb = ax.hexbin(df["carat"], df["price"], gridsize=45, bins="log", xscale="log", yscale="log", cmap="viridis")
    fig.colorbar(hb, ax=ax, label="diamonds per cell (log)")
    ax.set_title("2. Carat vs price (log-log)")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Price (US$)")

    # (3) the cut paradox: mean price AND mean carat per cut
    ax = axes[0, 2]
    by_cut = df.groupby("cut", observed=True).agg(mean_price=("price", "mean"), mean_carat=("carat", "mean"))
    ax.bar(by_cut.index.astype(str), by_cut["mean_price"], color="tab:blue", alpha=0.8)
    ax.set_ylabel("Mean price (US$)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(by_cut.index.astype(str), by_cut["mean_carat"], "o-", color="tab:red")
    ax2.set_ylabel("Mean carat", color="tab:red")
    ax2.grid(False)
    ax.set_title("3. Price and size by cut (worst -> best)")
    ax.set_xlabel("Cut")

    # (4) price spread by clarity
    ax = axes[1, 0]
    sns.boxplot(data=df, x="clarity", y="price", hue="clarity", order=CLARITY_ORDER, hue_order=CLARITY_ORDER,
                palette="crest", legend=False, fliersize=1, ax=ax)
    ax.set_yscale("log")
    ax.set_title("4. Price by clarity (log scale)")
    ax.set_xlabel("Clarity (worst -> best)")
    ax.set_ylabel("Price (US$)")

    # (5) average PRICE PER CARAT by colour x clarity (size removed from the picture)
    ax = axes[1, 1]
    ppc = df.assign(price_per_carat=df["price"] / df["carat"]).pivot_table(
        index="clarity", columns="color", values="price_per_carat", aggfunc="mean", observed=True
    ).reindex(index=CLARITY_ORDER[::-1], columns=COLOR_ORDER)
    sns.heatmap(ppc, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "US$ per carat"}, ax=ax)
    ax.set_title("5. Mean price per carat: clarity x colour")
    ax.set_xlabel("Colour (J = yellowest ... D = colourless)")
    ax.set_ylabel("Clarity")

    # (6) correlation heatmap of the numeric columns
    ax = axes[1, 2]
    sns.heatmap(df[NUMERIC_COLUMNS].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("6. Correlation of numeric columns")

    fig.suptitle(f"Diamonds dashboard  |  data: {source}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Insight
# --------------------------------------------------------------------------- #
def cut_paradox_insight(df: pd.DataFrame) -> str:
    """Compare raw average price by cut with price inside one narrow carat band."""
    by_cut = df.groupby("cut", observed=True).agg(price=("price", "mean"), carat=("carat", "mean"))
    worst, best = CUT_ORDER[0], CUT_ORDER[-1]
    band = df[(df["carat"] >= 0.9) & (df["carat"] <= 1.1)]
    in_band = band.groupby("cut", observed=True)["price"].mean()
    corr_carat_price = df["carat"].corr(df["price"])

    if by_cut.loc[best, "price"] < by_cut.loc[worst, "price"]:
        head = (
            f"The cut paradox: '{best}' diamonds (the best cut) have a LOWER average price (${by_cut.loc[best, 'price']:,.0f}) "
            f"than '{worst}' ones (${by_cut.loc[worst, 'price']:,.0f}) because they are much smaller "
            f"({by_cut.loc[best, 'carat']:.2f} vs {by_cut.loc[worst, 'carat']:.2f} carat on average)."
        )
    else:
        head = (
            f"Average price by cut is not ordered by quality (best cut '{best}': ${by_cut.loc[best, 'price']:,.0f}, "
            f"worst cut '{worst}': ${by_cut.loc[worst, 'price']:,.0f}) because the sizes differ "
            f"({by_cut.loc[best, 'carat']:.2f} vs {by_cut.loc[worst, 'carat']:.2f} carat)."
        )
    tail = (
        f" Carat alone correlates r={corr_carat_price:.2f} with price, so size hides everything else; comparing only "
        f"0.9-1.1 carat stones, '{best}' averages ${in_band.get(best, float('nan')):,.0f} vs "
        f"${in_band.get(worst, float('nan')):,.0f} for '{worst}'."
    )
    return head + tail


def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw, source, simulated = load_data(offline)
    df = prepare(raw)
    banner = "  *** SIMULATED DATA - run with internet for the real seaborn dataset ***" if simulated else ""
    print(f"[data] source = {source}{banner}")
    print(f"[data] {len(raw):,} rows loaded, {len(df):,} kept after removing zero-size diamonds\n")

    print("Summary statistics (numeric columns):")
    print(df[NUMERIC_COLUMNS].describe().round(2).T[["mean", "50%", "min", "max"]].to_string(), "\n")

    print("Mean price and size by cut:")
    print(df.groupby("cut", observed=True).agg(mean_price=("price", "mean"), mean_carat=("carat", "mean"),
                                               count=("price", "size")).round(2).to_string(), "\n")

    path = os.path.join(OUT_DIR, "diamonds_dashboard.png")
    build_dashboard(df, source, path)

    insight = cut_paradox_insight(df)
    print("KEY INSIGHT:", insight)
    print(f"\nDashboard saved to: {path}")
    return {"insight": insight, "source": source, "simulated": simulated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 05 - diamonds dashboard")
    parser.add_argument("--offline", action="store_true", help="use the simulated stand-in only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
