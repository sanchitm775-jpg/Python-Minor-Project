# Project 05 - Diamonds Price Dashboard (data visualization)
**Author:** Sanchit Kumar Mishra

### Problem Statement
Build one multi-chart dashboard that explains what drives diamond prices (carat, cut, colour, clarity) and avoids a classic misreading of the data.

### Dataset
- **Default:** seaborn's `diamonds` dataset (53,940 diamonds; price, carat, cut, colour, clarity, depth, table, x, y, z).
- **Fallback:** a **simulated** stand-in with the same columns (6,000 rows, fixed seed) so the script runs offline. The console and the chart title say **SIMULATED** whenever it is used. Its numbers are *not* real diamond statistics.

### Approach
Six panels in one figure (`outputs/diamonds_dashboard.png`): (1) price histogram on a log axis with mean vs median, (2) carat-vs-price hexbin on log-log axes, (3) mean price **and** mean carat by cut, (4) price by clarity boxplots, (5) mean *price per carat* heat-map for colour x clarity, (6) correlation heat-map. Zero-size rows in the real data are removed; cut/colour/clarity become ordered categories.

### Result
Verified run on the simulated stand-in: the dashboard renders all 6 panels and prints summary statistics plus the cut table below.

| Cut | Mean price | Mean carat |
|---|---|---|
| Fair | $5,660 | 1.17 |
| Good | $4,329 | 0.97 |
| Very Good | $3,364 | 0.83 |
| Premium | $4,046 | 0.90 |
| Ideal | $2,525 | 0.67 |

*I could not reach the real dataset from my build environment, so these specific numbers are simulated. Run it once with internet and the script prints the real ones.*

### Key Insight
The "cut paradox": best-cut (Ideal) diamonds average a *lower* price than worst-cut (Fair) ones because they are smaller - carat (r = 0.92 with price) hides everything else until you compare stones of equal size.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # downloads seaborn's dataset if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
