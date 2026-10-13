# Project 06 - Airline Passenger Forecasting (time series)
**Author:** Sanchit Kumar Mishra

### Problem Statement
Forecast monthly airline passengers with simple, explainable methods and decide whether the seasonal pattern is additive or multiplicative.

### Dataset
Monthly international airline passengers (thousands), Jan 1949 - Dec 1960 (144 values) - identical to `seaborn.load_dataset('flights')` (the classic *AirPassengers* series). It is **bundled** in `data/flights.csv` (real data, checked against the published mean 280.30 and total 40,363), so no download is needed.

### Approach
1. **Trend:** 12-month centred moving average, straight-line and exponential trend lines.
2. **Three forecasters:** seasonal naive (repeat last year); linear trend + additive seasonality; exponential trend + multiplicative seasonality (a linear trend fitted on log values).
3. **Evaluation:** rolling-origin backtest - forecast each of the last 5 years (1956-1960) using only earlier data, then average the errors. One test year can be a fluke.
4. Refit the best method on all data and forecast 1961.

### Result
Average over the 5 test years:

| Method | MAE | RMSE | MAPE |
|---|---|---|---|
| Exponential trend + multiplicative seasonality | 26.0 | 29.1 | **6.65 %** |
| Linear trend + additive seasonality | 30.0 | 39.8 | 6.97 % |
| Seasonal naive | 38.4 | 40.7 | 9.68 % |

Per-year MAPE winners: multiplicative in 1956 and 1957 (2.4 % / 3.2 % vs 7.8 % / 7.3 %), additive in 1958-1960.

### Key Insight
Seasonal swings grew 5.3x (44k -> 232k) but stayed about 43 % of each year's average, so seasonality is multiplicative - yet the best model flips from year to year, so a single-year backtest can mislead.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # no internet needed - data is bundled
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
