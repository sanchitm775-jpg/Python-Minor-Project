# Project 07 - Content-Based Book Recommender  *(uses a free API)*
**Author:** Sanchit Kumar Mishra

### Problem Statement
"If you liked this book, you might also like ..." - recommend similar books using only their subject tags (no user ratings needed).

### Dataset
- **Default:** the **Open Library Search API** (`https://openlibrary.org/search.json`, free, no key). For each of 6 genres (science fiction, fantasy, mystery, romance, horror, history) it fetches popular books and cleans their subject lists (drops catalogue noise like "Accessible book", very long phrases, and the genre label itself).
- **Fallback:** `data/books_sample.csv` - 51 well-known real books (e.g. Dune, Dracula, Pride and Prejudice) with subject tags I wrote in the style of Open Library headings.

### Approach
1. Turn each book's tags into a **TF-IDF** vector; similarity = **cosine similarity**; a book never recommends itself.
2. Evaluate honestly: hide the genre label from the model and measure how often the top-5 recommendations still share the book's genre ("genre purity"). Compare with random picks and with raw word counts (no IDF).
3. List the books whose neighbours stray furthest from their own genre.

### Result
Bundled sample (`--offline`), top-5 genre purity:

| Method | Purity |
|---|---|
| Random recommendations | 16.8 % |
| Raw word counts | 59.2 % |
| TF-IDF | 60.4 % |

Example: *Pride and Prejudice* -> Sense and Sensibility (0.84), Jane Eyre (0.54), Rebecca (0.44). TF-IDF and raw counts are about level on this small collection - the difference is within noise, so I do not claim TF-IDF "wins".

### Key Insight
Subject tags alone, with the genre label hidden, put 60 % of top-5 recommendations on the right genre shelf (random: 17 %), and the misses reveal cross-genre links - e.g. all 5 neighbours of *American Gods* are horror books.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # uses the Open Library API if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
Try your own book: `python main.py --title "Dracula"` (a partial title is fine).
