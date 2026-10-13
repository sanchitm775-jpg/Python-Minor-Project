"""
Project 07 - Content-Based Book Recommender  (uses a FREE API)
Author : Sanchit Kumar Mishra

Goal
----
"If you liked this book, you might also like ..."  Each book is described by its
subject tags; books with similar tags (cosine similarity on TF-IDF vectors) are
recommended.  No user ratings are needed - that is what "content-based" means.

Data
----
1. Default  : Open Library Search API  https://openlibrary.org/search.json
              (free, no API key) - ~25 popular books for each of 6 genres.
2. Fallback : data/books_sample.csv - 51 well-known books with subject tags
              (used when offline or with --offline).

Run
---
    python main.py                       # live API if reachable, else the sample
    python main.py --offline             # force the bundled sample
    python main.py --title "Dracula"     # recommendations for one book
"""
from __future__ import annotations

import argparse
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
SAMPLE_CSV = os.path.join(HERE, "data", "books_sample.csv")

OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "MastersLabProjects/1.0 (student project by Sanchit Kumar Mishra)"}
GENRES = ["science fiction", "fantasy", "mystery", "romance", "horror", "history"]
BOOKS_PER_GENRE = 25
# Open Library subjects that describe the *catalogue record*, not the book's content
NOISE_WORDS = ("accessible", "daisy", "large type", "in library", "internet archive", "overdrive",
               "reading level", "new york times", "bestseller", "nyt:", "lending", "open library",
               "protected", "juvenile", "english language", "fiction, general")


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def clean_subjects(subjects: list[str], genre: str, max_tags: int = 15) -> list[str]:
    """Lower-case, drop catalogue noise / very long phrases / the genre label itself."""
    tags, seen = [], set()
    for subject in subjects or []:
        tag = re.sub(r"\s+", " ", str(subject).lower().strip())
        if (not tag or len(tag.split()) > 4 or any(n in tag for n in NOISE_WORDS)
                or tag == genre or tag in seen):
            continue
        seen.add(tag)
        tags.append(tag)
    return tags[:max_tags]


def parse_search_results(payload: dict, genre: str, max_books: int = BOOKS_PER_GENRE) -> list[dict]:
    """Open Library search JSON -> list of {title, author, year, genre, tags} (most-published editions first)."""
    docs = sorted(payload.get("docs", []), key=lambda d: d.get("edition_count", 0), reverse=True)
    books = []
    for doc in docs:
        tags = clean_subjects(doc.get("subject", []), genre)
        if not doc.get("title") or len(tags) < 3:  # too little information to describe the book
            continue
        books.append({
            "title": doc["title"],
            "author": (doc.get("author_name") or ["Unknown"])[0],
            "year": doc.get("first_publish_year"),
            "genre": genre,
            "tags": "; ".join(tags),
        })
        if len(books) == max_books:
            break
    return books


def fetch_open_library() -> pd.DataFrame:
    """Query the Open Library search API once per genre."""
    books = []
    for genre in GENRES:
        response = requests.get(
            OPEN_LIBRARY_URL,
            params={"q": f'subject:"{genre}"', "limit": 100,
                    "fields": "title,author_name,first_publish_year,subject,edition_count"},
            headers=HEADERS, timeout=30,
        )
        response.raise_for_status()
        books.extend(parse_search_results(response.json(), genre))
        time.sleep(0.5)  # be polite to a free public API
    df = pd.DataFrame(books).drop_duplicates(subset=["title", "author"]).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("API returned too few usable books")
    return df


def load_data(offline: bool = False) -> tuple[pd.DataFrame, str]:
    if not offline:
        try:
            df = fetch_open_library()
            return df, f"Open Library API (live, {len(df)} books)"
        except Exception as err:
            print(f"[data] Open Library API unavailable ({type(err).__name__}) -> using bundled sample")
    df = pd.read_csv(SAMPLE_CSV)
    return df, f"bundled sample ({len(df)} well-known books)"


# --------------------------------------------------------------------------- #
# 2. Recommender
# --------------------------------------------------------------------------- #
class BookRecommender:
    """Content-based recommender: cosine similarity between books' tag vectors."""

    def __init__(self, books: pd.DataFrame, use_idf: bool = True):
        self.books = books.reset_index(drop=True)
        # Every tag text becomes a bag of words. TF-IDF down-weights words that appear in
        # almost every book (e.g. 'fiction'); plain counts treat all words equally.
        vectorizer_class = TfidfVectorizer if use_idf else CountVectorizer
        self.vectorizer = vectorizer_class(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.books["tags"].str.replace(";", " "))
        self.similarity = cosine_similarity(self.matrix)
        np.fill_diagonal(self.similarity, -1.0)  # a book must never recommend itself

    def find(self, title: str) -> int:
        """Index of the first book whose title contains `title` (case-insensitive)."""
        matches = self.books.index[self.books["title"].str.contains(re.escape(title), case=False)]
        if len(matches) == 0:
            raise KeyError(f"No book matching '{title}'. Try one of: {', '.join(self.books['title'].head(5))} ...")
        return int(matches[0])

    def recommend(self, title: str, k: int = 5) -> pd.DataFrame:
        """Return the k most similar books with their similarity score."""
        idx = self.find(title)
        top = np.argsort(self.similarity[idx])[::-1][:k]
        result = self.books.loc[top, ["title", "author", "genre"]].copy()
        result["similarity"] = self.similarity[idx, top].round(3)
        return result.reset_index(drop=True)

    def genre_purity(self, k: int = 5) -> pd.Series:
        """For every book: what share of its top-k recommendations has the SAME genre?

        The genre word itself was removed from the tags, so this tests whether the
        *other* subjects are enough to tell which shelf a book belongs on.
        """
        genres = self.books["genre"].to_numpy()
        purity = []
        for i in range(len(self.books)):
            top = np.argsort(self.similarity[i])[::-1][:k]
            purity.append(np.mean(genres[top] == genres[i]))
        return pd.Series(purity, index=self.books["title"])


# --------------------------------------------------------------------------- #
# 3. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False, title: str | None = None) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    books, source = load_data(offline)
    print(f"[data] source = {source}")
    print(f"[data] books per genre: {books['genre'].value_counts().to_dict()}\n")

    tfidf_model = BookRecommender(books, use_idf=True)
    count_model = BookRecommender(books, use_idf=False)

    # --- recommendations to show ---------------------------------------------
    wanted = [title] if title else ["Dune", "Pride and Prejudice", "Dracula"]
    shown = 0
    for name in wanted:
        try:
            idx = tfidf_model.find(name)
        except KeyError as err:
            print(f"  ({err})")
            continue
        book = books.loc[idx]
        print(f"If you liked '{book['title']}' by {book['author']} [{book['genre']}], try:")
        print(tfidf_model.recommend(name, k=5).to_string(index=False), "\n")
        shown += 1
    if shown == 0:  # none of the default titles exist in a live dataset -> use the first book
        first = books.loc[0, "title"]
        print(f"If you liked '{first}', try:")
        print(tfidf_model.recommend(first, k=5).to_string(index=False), "\n")

    # --- evaluation: genre purity with and without IDF ----------------------------
    purity_tfidf = tfidf_model.genre_purity(k=5)
    purity_count = count_model.genre_purity(k=5)
    chance = float((books["genre"].value_counts(normalize=True) ** 2).sum())  # prob. that a random pick matches
    summary = pd.Series({
        "Random recommendations": chance,
        "Raw word counts": purity_count.mean(),
        "TF-IDF": purity_tfidf.mean(),
    })
    print("Genre purity of the top-5 recommendations (genre label hidden from the model):")
    print((summary * 100).round(1).astype(str).add(" %").to_string(), "\n")

    # --- 'bridge' books: the ones whose neighbours mostly come from OTHER genres ------------
    bridge_notes = []
    for book_title in purity_tfidf.sort_values().head(4).index:
        idx = books.index[books["title"] == book_title][0]
        top_idx = np.argsort(tfidf_model.similarity[idx])[::-1][:5]
        counts = books.loc[top_idx, "genre"].value_counts()
        bridge_notes.append({"title": book_title, "genre": books.loc[idx, "genre"],
                             "top_genre": counts.index[0], "n": int(counts.iloc[0]), "all": counts.to_dict()})
    print("Books whose recommendations stray furthest from their own genre:")
    for note in bridge_notes:
        print(f"  {note['title']} [{note['genre']}] -> neighbours' genres: {note['all']}")
    print()

    # --- plot: similarity heatmap grouped by genre + purity bars -----------------------
    order = books.sort_values(["genre", "title"]).index
    sim_display = tfidf_model.similarity.copy()
    np.fill_diagonal(sim_display, 1.0)
    sim_display = sim_display[np.ix_(order, order)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.6, 1]})
    image = axes[0].imshow(sim_display, cmap="viridis", vmin=0, vmax=1)
    boundaries = np.cumsum(books.loc[order, "genre"].value_counts(sort=False).reindex(
        books.loc[order, "genre"].unique()).to_numpy())[:-1] - 0.5
    for b in boundaries:
        axes[0].axhline(b, color="white", lw=0.8)
        axes[0].axvline(b, color="white", lw=0.8)
    axes[0].set_title("Book-to-book similarity (books grouped by genre)")
    axes[0].set_xlabel("books")
    axes[0].set_ylabel("books")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    (summary * 100).plot.bar(ax=axes[1], color=["gray", "tab:orange", "tab:green"])
    axes[1].set_ylabel("Top-5 recommendations from the same genre (%)")
    axes[1].set_title("Does the genre label come back out of the tags?")
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle(f"Project 07 - book recommender | data: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "book_recommender.png"), dpi=120)
    plt.close(fig)

    # --- key insight (computed) ------------------------------------------------------------
    gain = (purity_tfidf.mean() - purity_count.mean()) * 100
    if gain >= 3:
        idf_sentence = f"TF-IDF weighting lifts it from {purity_count.mean():.0%} (raw counts) to {purity_tfidf.mean():.0%}"
    else:
        idf_sentence = (f"TF-IDF and raw counts end up about level ({purity_tfidf.mean():.0%} vs {purity_count.mean():.0%}) "
                        f"on this small collection")
    strays = "; ".join(f"{n['title']} ({n['genre']}) gets {n['n']} of 5 recommendations from {n['top_genre']}"
                       for n in bridge_notes[:2])
    insight = (
        f"With the genre label hidden, subject tags alone put {purity_tfidf.mean():.0%} of the top-5 recommendations on the "
        f"right genre shelf (random: {chance:.0%}); {idf_sentence}. The misses are informative, because shared tags cross "
        f"genre lines: {strays}."
    )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'book_recommender.png')}")
    return {"summary": summary, "insight": insight, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 07 - content-based book recommender")
    parser.add_argument("--offline", action="store_true", help="use the bundled sample only")
    parser.add_argument("--title", help="show recommendations for this book (partial title is fine)")
    args = parser.parse_args()
    run(offline=args.offline, title=args.title)


if __name__ == "__main__":
    main()
