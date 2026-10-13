# Class Notes - how to present each project
**Author:** Sanchit Kumar Mishra

For each project: a 30-second explanation, what to show live, and questions you may be asked (with short answers).
Tip: run with `--offline` in class so it works without Wi-Fi and the numbers match the README.

## Ideas that come up in several projects
| Term | One-line meaning |
|---|---|
| Cross-validation | Split the data into folds; train on some, test on the rest, rotate. Gives an honest score instead of testing on data the model has already seen. |
| Baseline | A dumb model (always predict the mean / the majority class). A real model must beat it. |
| R² | Share of the variation in the target the model explains (1 = perfect, 0 = no better than the mean). |
| Precision / recall | Precision: of the items I flagged, how many were right? Recall: of all the real ones, how many did I find? |
| Fallback data | Each project uses real data when the internet works and bundled data when it does not; the console always says which one is in use. |

---

## 01 - MPG regression
**30 seconds:** "I predict a car's mpg from cylinders, displacement, horsepower and weight, and compare six models with repeated cross-validation. The surprise is that *how you express the target* matters: modelling gallons per 100 miles instead of mpg makes a one-feature model almost as good as the four-feature one."
**Show:** the left chart (straight line on mpg vs curve from 100/mpg) and the results table.
**Q: Why not just trust R² on the training data?** It rewards memorising; cross-validated R² estimates performance on unseen cars.
**Q: Why is weight enough?** Displacement, horsepower and cylinders are strongly correlated with weight (r up to 0.89) - they carry mostly the same information (multicollinearity).
**Q: Why only 32 cars?** That is the offline fallback (*mtcars*). With internet the script uses seaborn's 398-car `mpg` dataset.

## 02 - Wine classification
**30 seconds:** "Three cultivars, 13 chemical measurements. Most models reach about 98 %, but kNN drops to 70 % unless features are scaled, because one column (`proline`, values up to ~1,700) holds 99.8 % of the variance."
**Show:** the printed model table - the raw-vs-scaled kNN rows are the story.
**Q: Why does kNN need scaling but a random forest does not?** kNN uses distances, so big-number columns dominate; trees split on one feature at a time and ignore scale.
**Q: What is stratified cross-validation?** Each fold keeps the same class proportions as the full data.

## 03 - Country clustering (World Bank API)
**30 seconds:** "I pull four indicators for every country from the free World Bank API and cluster them with K-Means. k is chosen by silhouette score. Then I ask which countries are healthier or sicker than their income predicts."
**Show:** the PCA scatter and the income-vs-life-expectancy chart.
**Q: Why log GDP?** Income is hugely skewed (hundreds to tens of thousands of dollars); the log makes differences comparable.
**Q: Why scale before K-Means?** K-Means uses Euclidean distance, so features must be on comparable scales.
**Q: What is the silhouette score?** For each point: how close it is to its own cluster compared to the nearest other cluster (-1 to 1; higher is better).
**Q: What assumption did you make?** Missing literacy for high-income countries is set to 99 %; offline mode uses approximate numbers.

## 04 - Spam classifier
**30 seconds:** "TF-IDF turns each message into numbers; Naive Bayes and logistic regression classify it. The lesson is metrics: always answering 'ham' is already about 68 % accurate on my sample and finds no spam, so I report precision and recall for the spam class."
**Show:** the four live predictions - note that "are you *free* for lunch" is correctly ham.
**Q: What is TF-IDF?** Word counts, down-weighted for words that appear in almost every message.
**Q: Why replace links and long numbers with `urltoken` / `longnumber`?** So the model learns "has a link / phone number" rather than memorising each unique URL.
**Q: Is the sample real?** No - it is 164 messages I wrote; the script downloads the real UCI dataset when online.

## 05 - Diamonds dashboard
**30 seconds:** "Six linked charts explain diamond prices. The key trap is the 'cut paradox': the best cut appears *cheaper* than the worst because better-cut stones are smaller, and carat dominates price."
**Show:** panel 3 (bars = price, line = carat) next to panel 2.
**Q: Why log axes?** Price and carat are right-skewed and the relationship is roughly a power law; log-log makes it near-linear.
**Q: Is it real data?** Real seaborn data when online; otherwise a simulated stand-in clearly labelled in the title.

## 06 - Flights forecasting
**30 seconds:** "Monthly passengers 1949-60. I fit a trend and seasonality three ways and test each on five different years using only earlier data. Seasonal swings grow with the level, so multiplicative seasonality makes sense - but the winner changes by year, so one backtest year is not enough."
**Show:** chart 4 (swing grows) and the per-year MAPE table.
**Q: Additive vs multiplicative seasonality?** Additive: the seasonal bump is a fixed number of passengers. Multiplicative: it is a fixed percentage of the level.
**Q: Why not random train/test split?** Time order matters; you must never train on the future.
**Q: What is MAPE?** Mean absolute percentage error.

## 07 - Book recommender (Open Library API)
**30 seconds:** "Each book is a bag of subject tags; TF-IDF + cosine similarity finds the nearest books. To test it fairly I hide the genre label and check whether recommendations still land on the right shelf: 60 % vs 17 % by chance."
**Show:** `python main.py --title "Dracula"` and the similarity heat-map (blocks along the diagonal = genres).
**Q: Content-based vs collaborative filtering?** Content-based uses item features; collaborative uses other users' behaviour (needs ratings).
**Q: Cold start?** A new book can be recommended immediately from its tags - an advantage of content-based methods.
**Q: Did TF-IDF beat raw counts?** On this small sample the difference is within noise, so I do not claim it.

## 08 - Wikipedia scraper
**30 seconds:** "I fetch the page with `requests`, parse it with BeautifulSoup, expand rowspans and colspans, strip citation marks, convert text to numbers and analyse. If the live page is unreachable or changes, the same parser runs on a bundled copy."
**Show:** `outputs/states_clean.csv` and the "first 8 rows" printout.
**Q: Is scraping allowed?** Check the site's terms and robots.txt; use a descriptive User-Agent, make few requests, prefer an official API where one exists (Wikipedia has one).
**Q: Why not `pandas.read_html`?** It is shorter, but the assignment asks for requests + BeautifulSoup, and writing the parser shows how HTML tables really work.

## 09 - File organizer
**30 seconds:** "No ML: a safe automation tool. It finds duplicates by SHA-256 hash after a cheap size filter, sorts files by type and month, never deletes, and can undo itself."
**Show:** `python main.py` - the demo prints the plan, the tree, and the verified undo.
**Q: Why hash only same-size files?** Different sizes cannot be identical - this skips most files.
**Q: Why not compare names?** Copies get renamed ("IMG (1).jpg"); two different files can share a name.
**Q: What if hashes collide?** SHA-256 collisions are practically impossible for this purpose.

## 10 - Weather EDA (Open-Meteo API)
**30 seconds:** "A step-by-step EDA: structure, missing values, statistics, outliers, correlation, seasonality. The main lesson is that rule-based outlier detection can be wrong: the IQR rule flags a quarter of days as rainy 'outliers', but they are the monsoon."
**Show:** panels 2 (boxplots), 3 (rain histogram) and 6 (rain vs temperature by month).
**Q: Pearson vs Spearman?** Pearson measures linear relationships; Spearman works on ranks and catches monotonic but non-linear ones.
**Q: Why not delete the outliers?** They are valid measurements (checked: most fall in the monsoon months).
**Q: Is the data real?** The API data is real; the offline stand-in is simulated and labelled as such.
