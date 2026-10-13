# Project 01 - Fuel Efficiency (MPG) Regression
**Author:** Sanchit Kumar Mishra

### Problem Statement
Predict how many miles per gallon (mpg) a car achieves from its engine specs (cylinders, displacement, horsepower, weight), and find out which spec really matters.

### Dataset
- **Default:** seaborn's `mpg` dataset (398 cars, 1970-82) - downloaded by seaborn on first use.
- **Fallback (used automatically offline or with `--offline`):** the classic 1974 *mtcars* table, 32 cars (`data/mtcars_fuel_efficiency.csv`). Real data; I checked its means, standard deviation and correlations against the published values.
- Common columns used in both: `cylinders, displacement, horsepower, weight (lbs)` -> target `mpg`.

### Approach
1. Correlation of each spec with mpg.
2. Compare 6 models with **5-fold cross-validation repeated 10 times**: mean baseline, linear regression (weight only / all specs), Ridge, random forest, and a linear model that predicts **100/mpg** (gallons per 100 miles) and converts back.
3. Scores are computed on pooled out-of-fold predictions, because scoring each tiny fold separately gives misleading R² on small data (the baseline came out at -0.53 before I fixed this).

### Result
Cross-validated, bundled mtcars data (`--offline`):

| Model | R² | RMSE (mpg) |
|---|---|---|
| Random forest, 4 specs | 0.830 | 2.45 |
| Ridge (scaled), 4 specs | 0.797 | 2.67 |
| Linear regression, 4 specs | 0.793 | 2.70 |
| Linear on 100/mpg, weight only | 0.769 | 2.85 |
| Linear regression, weight only | 0.691 | 3.29 |
| Baseline (predict the mean) | -0.084 | 6.18 |

Correlation with mpg: weight -0.87, cylinders -0.85, displacement -0.85, horsepower -0.78.
*Numbers come from the 32-car fallback; with the 398-car seaborn data the script prints its own figures.*

### Key Insight
Predicting fuel burned per distance (100/mpg) instead of mpg lifts a weight-only model from R² 0.69 to 0.77 - nearly matching all four specs (0.79) - because displacement, horsepower and cylinders mostly repeat what weight already says.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # tries the live source first, falls back automatically
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
