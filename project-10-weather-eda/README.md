# Project 10 - Systematic EDA of Daily Weather  *(uses a free API)*
**Author:** Sanchit Kumar Mishra

### Problem Statement
Run a complete exploratory data analysis - structure, missing values, summary statistics, outliers, correlations, seasonality - on five years of daily weather for Raipur, Chhattisgarh, and learn where each standard EDA rule can mislead.

### Dataset
- **Default:** the **Open-Meteo Historical Weather API** (`archive-api.open-meteo.com`, free, no key): daily max/min/mean temperature, rainfall, max wind speed, solar radiation and evaporation for 2019-01-01 to 2023-12-31 (1,826 days). API `null` values are kept as missing, not hidden.
- **Fallback:** a **simulated** stand-in with the same columns and a Raipur-like monsoon climate (fixed seed). The console and chart title say **SIMULATED** whenever it is used; its numbers are not real measurements.

### Approach
Six numbered steps, each printed and plotted (`outputs/weather_eda.png`): (1) structure and date continuity, (2) missing values, (3) summary statistics + skewness, (4) outliers with two rules (1.5 x IQR and |z| > 3) plus a physical sanity check (min temp <= max temp), (5) Pearson **and** Spearman correlation heat-maps, (6) monthly climatology. The final step checks *when* the flagged "outliers" occur instead of assuming they are errors.

### Result
Verified run on the simulated stand-in: 1,826 days, no duplicates or gaps, no missing values; rain skew 5.6 (mean 3.6 mm vs median 0 mm; 74 % of days are dry); IQR rule flags 24 % of days as rain outliers vs only 2.8 % by the z-score rule. Strongest correlations: the three temperature columns (~0.97-0.98), radiation~evaporation (0.86), rain~radiation (-0.71).

*I could not reach the live API from my build environment; run with internet and the script prints the real figures.*

### Key Insight
The IQR rule flags 24 % of days as rain "outliers" (Q3 ≈ 0.1 mm), but 86 % of them fall in June-September - they are the monsoon itself, so "removing outliers" would delete the signal.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # uses the Open-Meteo API if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
