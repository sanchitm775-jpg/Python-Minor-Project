# Project 03 - Segmenting Countries by Socio-economic Indicators  *(uses a free API)*
**Author:** Sanchit Kumar Mishra

### Problem Statement
Group countries into development tiers using GDP per capita, life expectancy, adult literacy and infant mortality - and see where income alone gives a misleading picture of well-being.

### Dataset
- **Default:** the **World Bank Open Data API** (`https://api.worldbank.org/v2`, free, no key). The script downloads 4 indicators for all countries (latest value in a recent window), removes regional aggregates such as "World", and treats a missing literacy figure for *High-income* countries as 99 % (documented assumption, because rich countries often do not report it).
- **Fallback:** `data/countries_fallback.csv` - 65 countries with **rounded, approximate** values (~2021) typed in by me for offline use. They are *not* exact World Bank figures; use the live API for real numbers.

### Approach
1. Log-transform GDP (income is extremely right-skewed) and standardise all features.
2. K-Means with k = 3...7, choose k by **silhouette score**.
3. Profile and name the tiers; view them with a 2-D PCA projection.
4. Fit life expectancy ~ log GDP and look at the residuals (who is healthier / less healthy than their income predicts).
5. Compare with a GDP-only clustering using the adjusted Rand index.

### Result
Bundled fallback data (`--offline`), best k = 3 (silhouette 0.47):

| Tier | Countries | Median GDP pc | Life exp. | Literacy | Infant mortality |
|---|---|---|---|---|---|
| 1 | 10 | $805 | 60.2 | 51 % | 54 |
| 2 | 23 | $2,500 | 67.7 | 83 % | 25 |
| 3 | 32 | $21,850 | 78.5 | 98 % | 5.4 |

### Key Insight
Income explains only part of health (log-GDP gives R² = 0.74 for life expectancy): Nigeria, Botswana and South Africa live 10-14 years less than their income predicts, while Algeria and Sri Lanka live 5-7 years more. *(from the approximate fallback data - the script recomputes this from live data)*

### How to Run
```bash
pip install -r requirements.txt
python main.py            # uses the World Bank API if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
