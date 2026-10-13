"""
Project 08 - Web Scraping a Wikipedia Table  (requests + BeautifulSoup)
Author : Sanchit Kumar Mishra

Goal
----
Scrape the table "List of states and union territories of India by population"
from Wikipedia, clean it into a tidy pandas DataFrame (numbers as numbers, no
citation marks like [1]) and analyse it.

Data
----
1. Default  : the live Wikipedia article (fetched with `requests`, parsed with BeautifulSoup).
2. Fallback : data/sample_page.html - an offline HTML page built like a Wikipedia article
              (multi-row header, rowspans, citation marks, a decoy table, a total row)
              containing Census-2011 figures.  The SAME parser is used for both, so the
              fallback also proves that the scraping code works.

Run
---
    python main.py            # live page if reachable, else the bundled HTML
    python main.py --offline  # force the bundled HTML
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
SAMPLE_HTML = os.path.join(HERE, "data", "sample_page.html")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_states_and_union_territories_of_India_by_population"
HEADERS = {"User-Agent": "MastersLabProjects/1.0 (student project by Sanchit Kumar Mishra; educational scraping)"}

# canonical column -> (header words we accept, header words we reject)
COLUMN_RULES = {
    "state": (("state", "union territory", "name"), ()),
    "population": (("population",), ("density", "growth", "percent", "%", "rural", "urban", "share")),
    "area_km2": (("area",), ("rank",)),
    "sex_ratio": (("sex ratio", "sex-ratio"), ()),
}


# --------------------------------------------------------------------------- #
# 1. Fetching
# --------------------------------------------------------------------------- #
def fetch_live_html() -> str:
    """Download the Wikipedia article (raises an exception when offline)."""
    response = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


# --------------------------------------------------------------------------- #
# 2. Parsing: HTML table -> list of rows
# --------------------------------------------------------------------------- #
def cell_text(cell) -> str:
    """Visible text of one <td>/<th>: no citation marks, no hidden sort keys, no footnotes."""
    for tag in cell.find_all(["sup", "style", "script"]):
        tag.decompose()  # <sup> holds citation marks like [1] and unit superscripts
    for tag in cell.find_all(style=re.compile(r"display\s*:\s*none")):
        tag.decompose()  # Wikipedia hides sort keys in invisible spans
    text = cell.get_text(" ", strip=True).replace("\xa0", " ")
    text = re.sub(r"\[[^\]]*\]", "", text)  # any leftover [1], [a], [note 2]
    return re.sub(r"\s+", " ", text).strip()


def table_to_grid(table) -> tuple[list[list[str]], int]:
    """Turn an HTML table into a rectangular list of rows, expanding rowspan/colspan.

    Returns (rows, number_of_header_rows).  Header rows are the leading rows whose
    cells are all <th>.
    """
    grid, header_flags = [], []
    carried: dict[tuple[int, int], str] = {}  # (row, col) -> text that a rowspan pushes into later rows
    for r, tr in enumerate(table.find_all("tr")):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row, col = [], 0
        for cell in cells:
            while (r, col) in carried:  # a cell from a rowspan above occupies this column
                row.append(carried.pop((r, col)))
                col += 1
            text = cell_text(cell)
            rowspan, colspan = int(cell.get("rowspan", 1) or 1), int(cell.get("colspan", 1) or 1)
            for _ in range(colspan):
                row.append(text)
                for extra in range(1, rowspan):
                    carried[(r + extra, col)] = text
                col += 1
        while (r, col) in carried:
            row.append(carried.pop((r, col)))
            col += 1
        grid.append(row)
        header_flags.append(all(c.name == "th" for c in cells))

    n_header = 0
    for flag in header_flags:  # count the leading all-<th> rows
        if not flag:
            break
        n_header += 1
    width = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (width - len(r)) for r in grid]
    return grid, n_header


def flatten_headers(header_rows: list[list[str]]) -> list[str]:
    """Merge multi-row headers: ['Census 2011', 'Population'] -> 'Census 2011 Population'."""
    names = []
    for col in range(len(header_rows[0])):
        parts = []
        for row in header_rows:
            if row[col] and row[col] not in parts:
                parts.append(row[col])
        names.append(" ".join(parts))
    return names


def map_columns(headers: list[str]) -> dict[str, int]:
    """Find which table column is the state name, population, area, sex ratio."""
    mapping: dict[str, int] = {}
    for key, (accept, reject) in COLUMN_RULES.items():
        for i, header in enumerate(headers):
            h = header.lower()
            if i in mapping.values() or any(word in h for word in reject):
                continue
            if any(word in h for word in accept):
                mapping[key] = i
                break
    return mapping


def to_number(text: str) -> float:
    """'1,210,854,977' -> 1210854977.0 ; '12.5%' -> 12.5 ; '-' or '' -> NaN."""
    match = re.search(r"-?\d[\d,]*\.?\d*", str(text))
    return float(match.group().replace(",", "")) if match else np.nan


def html_to_dataframe(html: str) -> pd.DataFrame:
    """Find the right table in the page and return a clean DataFrame.

    The page may contain several tables, so we pick the first <table class="wikitable">
    that has both a 'state' column and a 'population' column.
    """
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table", class_="wikitable"):
        grid, n_header = table_to_grid(table)
        if n_header == 0 or len(grid) - n_header < 10:
            continue
        headers = flatten_headers(grid[:n_header])
        mapping = map_columns(headers)
        if "state" not in mapping or "population" not in mapping:
            continue

        records = []
        for row in grid[n_header:]:
            record = {"state": re.sub(r"[*†‡]+$", "", row[mapping["state"]]).strip()}
            for key in ("population", "area_km2", "sex_ratio"):
                record[key] = to_number(row[mapping[key]]) if key in mapping else np.nan
            records.append(record)
        df = pd.DataFrame(records)
        df = df[df["state"].ne("") & ~df["state"].str.lower().str.startswith(("total", "india"))]
        df = df.dropna(subset=["population"]).reset_index(drop=True)
        if len(df) >= 10:
            return df
    raise ValueError("no suitable table found in the page")


def load_table(offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Try the live page, otherwise parse the bundled HTML with the very same function."""
    if not offline:
        try:
            return html_to_dataframe(fetch_live_html()), "live Wikipedia page"
        except Exception as err:
            print(f"[data] live Wikipedia scrape failed ({type(err).__name__}: {err}) -> using bundled HTML page")
    with open(SAMPLE_HTML, encoding="utf-8") as handle:
        return html_to_dataframe(handle.read()), "bundled sample page (Census 2011 figures)"


# --------------------------------------------------------------------------- #
# 3. Analysis
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, source = load_table(offline)
    df = df.sort_values("population", ascending=False).reset_index(drop=True)
    print(f"[data] source = {source}")
    print(f"[data] scraped and cleaned {len(df)} rows; columns = {list(df.columns)}")
    print(f"[data] missing values per column: {df.isna().sum().to_dict()}\n")
    print("First 8 rows of the clean table:")
    print(df.head(8).to_string(index=False), "\n")
    df.to_csv(os.path.join(OUT_DIR, "states_clean.csv"), index=False)

    total = df["population"].sum()
    df["share_%"] = df["population"] / total * 100
    df["cumulative_share_%"] = df["share_%"].cumsum()
    n_for_half = int((df["cumulative_share_%"] < 50).sum() + 1)  # smallest group of states holding >= 50 %
    top_names = ", ".join(df["state"].head(n_for_half))
    print(f"Total population in table: {total:,.0f}")
    print(f"{n_for_half} of {len(df)} states/UTs hold {df['cumulative_share_%'].iloc[n_for_half - 1]:.1f}% of it: {top_names}\n")

    insight_parts = [
        f"Just {n_for_half} of the {len(df)} states/UTs ({top_names}) hold "
        f"{df['cumulative_share_%'].iloc[n_for_half - 1]:.0f}% of the population."
    ]

    has_area = df["area_km2"].notna().sum() >= 10
    has_sex = df["sex_ratio"].notna().sum() >= 10
    if has_area:
        df["density_per_km2"] = df["population"] / df["area_km2"]
        national_density = total / df["area_km2"].sum()
        densest = df.sort_values("density_per_km2", ascending=False).head(3)
        print(f"National density = {national_density:,.0f} people/km2. Densest: " +
              ", ".join(f"{r.state} ({r.density_per_km2:,.0f})" for r in densest.itertuples()))
        biggest = df.iloc[0]
        insight_parts.append(
            f"Population rank is not density rank: {biggest['state']} has the most people but a density of "
            f"{biggest['density_per_km2']:,.0f}/km2, while {densest.iloc[0]['state']} packs "
            f"{densest.iloc[0]['density_per_km2']:,.0f}/km2 ({densest.iloc[0]['density_per_km2'] / national_density:.0f}x the national average)."
        )
    if has_area and has_sex:
        rho = df["density_per_km2"].rank().corr(df["sex_ratio"].rank())  # Spearman correlation
        lowest = df.sort_values("sex_ratio").head(3)
        print(f"Spearman correlation (density vs sex ratio) = {rho:.2f}")
        print("Lowest sex ratios: " + ", ".join(f"{r.state} ({r.sex_ratio:.0f})" for r in lowest.itertuples()), "\n")
        insight_parts.append(
            f"Sex ratio (females per 1,000 males) is lowest in {', '.join(lowest['state'])} "
            f"({lowest['sex_ratio'].min():.0f}-{lowest['sex_ratio'].max():.0f}); "
            f"{'density does not explain it: the' if abs(rho) < 0.3 else 'density is related to it: the'} "
            f"density-vs-sex-ratio rank correlation across all rows is {rho:.2f} "
            f"({'weak' if abs(rho) < 0.3 else 'moderate' if abs(rho) < 0.6 else 'strong'})."
        )

    # --- plots ---------------------------------------------------------------------
    n_plots = 2 if (has_area and has_sex) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7.5 * n_plots, 6), squeeze=False)
    top10 = df.head(10).iloc[::-1]
    axes[0, 0].barh(top10["state"], top10["population"] / 1e6, color="tab:blue")
    axes[0, 0].set_xlabel("Population (millions)")
    axes[0, 0].set_title("10 most populous states/UTs")
    if n_plots == 2:
        ax = axes[0, 1]
        ax.scatter(df["density_per_km2"], df["sex_ratio"], s=df["population"] / df["population"].max() * 400 + 15, alpha=0.6)
        ax.set_xscale("log")
        ax.axhline(1000, color="gray", ls=":")
        for r in df.sort_values("sex_ratio").head(3).itertuples():
            ax.annotate(r.state, (r.density_per_km2, r.sex_ratio), fontsize=8)
        ax.annotate(df.sort_values("sex_ratio").iloc[-1]["state"],
                    (df.sort_values("sex_ratio").iloc[-1]["density_per_km2"], df["sex_ratio"].max()), fontsize=8)
        ax.set_xlabel("Population density (people per km2, log scale)")
        ax.set_ylabel("Sex ratio (females per 1,000 males)")
        ax.set_title("Density vs sex ratio (bubble size = population)")
    fig.suptitle(f"Project 08 - scraped table | data: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "scraped_table_analysis.png"), dpi=120)
    plt.close(fig)

    insight = " ".join(insight_parts)
    print("KEY INSIGHT:", insight)
    print(f"\nClean table saved to: {os.path.join(OUT_DIR, 'states_clean.csv')}")
    return {"df": df, "insight": insight, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 08 - Wikipedia table scraper")
    parser.add_argument("--offline", action="store_true", help="parse the bundled HTML page only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
