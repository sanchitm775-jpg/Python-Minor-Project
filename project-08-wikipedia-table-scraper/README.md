# Project 08 - Web Scraping a Wikipedia Table
**Author:** Sanchit Kumar Mishra

### Problem Statement
Scrape "List of states and union territories of India by population" from Wikipedia with `requests` + `BeautifulSoup`, clean it into a tidy DataFrame, and analyse it.

### Dataset
- **Default:** the live Wikipedia article (`en.wikipedia.org/wiki/List_of_states_and_union_territories_of_India_by_population`).
- **Fallback:** `data/sample_page.html` - an offline HTML page built like a Wikipedia article (two-row header with `rowspan`/`colspan`, citation marks like `[1]`, hidden sort keys, a decoy table, a "Total" row) holding **Census-2011** figures for 36 states/UTs. Its population column adds up exactly to the census total (1,210,854,977). Because the *same parser* handles both pages, the fallback also proves that the scraper works.

### Approach
1. Fetch the page with a descriptive `User-Agent`.
2. Find the right table by its headers (needs a *state* and a *population* column), expand `rowspan`/`colspan` into a rectangular grid, merge multi-row headers.
3. Clean cells: remove citation marks and hidden spans, strip footnote symbols, convert `"1,210,854,977"` to numbers, drop the total row.
4. If the live page changes and no suitable table is found, fall back to the bundled page instead of crashing.
5. Analyse: population share, density (population / area), sex ratio.

### Result
36 rows x 4 columns (`state, population, area_km2, sex_ratio`), 0 missing values; saved to `outputs/states_clean.csv`. Top rows: Uttar Pradesh 199.8 M, Maharashtra 112.4 M, Bihar 104.1 M. National density 368 people/km²; densest: Delhi 11,313, Chandigarh 9,258, Puducherry 2,605.

### Key Insight
Just 6 of 36 states/UTs hold 54 % of the population, and the sex ratio is not explained by density (rank correlation -0.17) - the lowest ratios are in tiny UTs (Daman and Diu 618, Dadra and Nagar Haveli 774).

### How to Run
```bash
pip install -r requirements.txt
python main.py            # scrapes Wikipedia if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
