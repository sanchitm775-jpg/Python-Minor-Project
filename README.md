# 10 Python / ML Mini-Projects
**Author:** Sanchit Kumar Mishra  |  Master's lab assignment

Ten self-contained mini-projects covering regression, classification, clustering, NLP, visualization, time series, recommendation, web scraping, automation and EDA. Every project has its own `main.py`, `README.md` and `requirements.txt`, and prints a **KEY INSIGHT** computed from its data.

## Folder tree
```
.
├── README.md                      <- you are here
├── CLASS_NOTES.md                 <- how to present each project + likely questions
├── requirements.txt               <- installs everything for all projects at once
├── run_all.py                     <- runs all 10 projects and prints PASS/FAIL
├── .gitignore
├── tests/
│   └── test_live_paths_with_mocks.py   <- tests the API/download code with faked responses
├── project-01-mpg-regression/          main.py  README.md  requirements.txt  data/mtcars_fuel_efficiency.csv
├── project-02-wine-classification/     main.py  README.md  requirements.txt
├── project-03-country-clustering/      main.py  README.md  requirements.txt  data/countries_fallback.csv
├── project-04-spam-classifier/         main.py  README.md  requirements.txt  data/sms_sample.csv
├── project-05-diamonds-dashboard/      main.py  README.md  requirements.txt
├── project-06-flights-forecasting/     main.py  README.md  requirements.txt  data/flights.csv
├── project-07-book-recommender/        main.py  README.md  requirements.txt  data/books_sample.csv
├── project-08-wikipedia-table-scraper/ main.py  README.md  requirements.txt  data/sample_page.html
├── project-09-file-organizer/          main.py  README.md  requirements.txt
└── project-10-weather-eda/             main.py  README.md  requirements.txt
```

## The projects
| # | Category | Topic | Data source | Free API? |
|---|---|---|---|---|
| 01 | Regression | Fuel efficiency (MPG) from engine specs | seaborn `mpg` -> bundled mtcars | |
| 02 | Classification | Wine cultivar from chemistry | `sklearn.load_wine` (built in) | |
| 03 | Clustering | Countries by GDP, life expectancy, literacy, infant mortality | **World Bank API** -> bundled snapshot | **Yes** |
| 04 | NLP | Spam vs ham SMS | UCI SMS Spam Collection -> bundled sample | |
| 05 | Visualization | Diamonds price dashboard (6 charts) | seaborn `diamonds` -> simulated stand-in | |
| 06 | Time series | Airline passengers forecast | bundled AirPassengers (= seaborn `flights`) | |
| 07 | Recommendation | Content-based book recommender | **Open Library API** -> bundled sample | **Yes** |
| 08 | Web scraping | Wikipedia table: Indian states by population | Wikipedia (`requests` + BeautifulSoup) -> bundled HTML | |
| 09 | Automation | Duplicate finder + file organizer (no ML) | your local folders / built-in demo | |
| 10 | EDA | Daily weather in Raipur, 2019-2023 | **Open-Meteo API** -> simulated stand-in | **Yes** |

Projects 03, 07 and 10 use free APIs that need **no API key**.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or install one project's requirements.txt
```
Tested with Python 3.12, pandas 3.0, NumPy 2.4, scikit-learn 1.8, matplotlib 3.10, seaborn 0.13. The code avoids version-specific features and the `requirements.txt` files set lower bounds, but I could only run it on those versions.

## Run
```bash
cd project-01-mpg-regression && python main.py        # one project
python run_all.py                                      # all 10, offline/bundled data (reproducible)
python run_all.py --live                               # all 10, real sources first (needs internet)
python -m unittest tests.test_live_paths_with_mocks -v # tests for the API/download code
```
Every project accepts `--offline`, which forces the bundled data. Use it in a classroom without Wi-Fi: the numbers then match each project's README exactly. Plots and tables land in each project's `outputs/` folder.

## How the data fallbacks work (please read before presenting)
Each project that needs the internet tries the real source first and **falls back automatically**, printing which data it used (`[data] source = ...`). The fallbacks are *not* all equal:

| Project | Fallback is... |
|---|---|
| 02, 06, 09 | No fallback needed - real built-in/bundled data or no data |
| 01 | **Real** (mtcars, 32 cars) but a different, smaller dataset than seaborn's `mpg` |
| 08 | Census-2011 figures in a Wikipedia-style HTML page; same parser as the live page |
| 03 | **Approximate** rounded values typed in by me - for demonstration only |
| 04 | **Hand-written** sample messages in the style of the real data - not the UCI data |
| 07 | Real book titles with tags I wrote in the style of Open Library headings |
| 05, 10 | **Simulated** data; clearly labelled `SIMULATED` in the console and on the charts |

The numbers quoted in each project's README were produced from these bundled/fallback datasets (`--offline`). When you run with internet, the script prints the numbers for the real data - and you should re-run once and update the README if you want it to quote those instead.

## Verification status
- All 10 `main.py` scripts ran end-to-end without errors in offline mode and in live mode with automatic fallback (`python run_all.py`, `python run_all.py --live`).
- 11 unit tests exercise the live-data code (World Bank, Open Library, Open-Meteo, UCI download, Wikipedia HTML, seaborn loaders) against **mocked** responses in each service's documented format.
- **Not verified:** actual connections to the live services. My build environment had no internet access. Run `python run_all.py --live` once on your machine to confirm.

## Put it on GitHub
```bash
git init
git add .
git commit -m "Add 10 Python/ML mini-projects"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git   # create the empty repo on github.com first
git push -u origin main
```
