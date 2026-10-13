"""
Project 04 - Spam vs Ham SMS Classification  (NLP basics)
Author : Sanchit Kumar Mishra

Goal
----
Classify a text message as 'spam' or 'ham' (= a normal, wanted message) using
TF-IDF word features and two classic models (Naive Bayes, Logistic Regression).

Data
----
1. Default  : the UCI "SMS Spam Collection" (5,574 real messages), downloaded
              from the UCI repository (or a public mirror of the same file).
2. Fallback : data/sms_sample.csv - 164 messages WRITTEN BY ME in the style of
              the real data (53 spam / 111 ham).  It is NOT the UCI data; it only
              exists so the script runs offline and in a classroom without Wi-Fi.

Run
---
    python main.py            # real data if reachable, else the sample
    python main.py --offline  # force the sample
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
SAMPLE_CSV = os.path.join(HERE, "data", "sms_sample.csv")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_TSV_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
HEADERS = {"User-Agent": "MastersLabProjects/1.0 (student project by Sanchit Kumar Mishra)"}


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def parse_sms_tsv(text: str) -> pd.DataFrame:
    """Parse the 'label<TAB>message' format used by the UCI file and its mirrors."""
    df = pd.read_csv(
        io.StringIO(text), sep="\t", header=None, names=["label", "text"],
        quoting=csv.QUOTE_NONE,  # messages contain quote characters - do not treat them as CSV quotes
    )
    df = df[df["label"].isin(["ham", "spam"])].dropna().drop_duplicates().reset_index(drop=True)
    return df


def download_real_data() -> pd.DataFrame:
    """Try the UCI zip first, then a GitHub mirror of the same dataset."""
    try:
        response = requests.get(UCI_ZIP_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            text = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
        return parse_sms_tsv(text)
    except Exception:
        response = requests.get(MIRROR_TSV_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return parse_sms_tsv(response.text)


def load_data(offline: bool = False) -> tuple[pd.DataFrame, str]:
    if not offline:
        try:
            df = download_real_data()
            return df, f"UCI SMS Spam Collection ({len(df)} real messages)"
        except Exception as err:
            print(f"[data] could not download the UCI data ({type(err).__name__}) -> using bundled sample")
    df = pd.read_csv(SAMPLE_CSV)
    return df, f"bundled hand-written SAMPLE ({len(df)} messages, NOT the real UCI data)"


# --------------------------------------------------------------------------- #
# 2. Simple hand-made statistics (to understand the data before modelling)
# --------------------------------------------------------------------------- #
def message_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Average length, digit count and share of CAPITAL letters for each class."""
    stats = pd.DataFrame({
        "label": df["label"],
        "characters": df["text"].str.len(),
        "digits": df["text"].str.count(r"\d"),
        "capital_share": df["text"].apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)),
    })
    return stats.groupby("label").mean().round(3)


# --------------------------------------------------------------------------- #
# 3. Modelling
# --------------------------------------------------------------------------- #
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
LONG_NUMBER_PATTERN = re.compile(r"(?:\d-?){5,}")  # 5+ digits in a row: phone numbers, short codes, prizes


def normalise(text: str) -> str:
    """Lower-case the text and replace every link / long number with one shared word.

    Without this, each fake URL or phone number would be a brand-new word that the
    model has never seen.  With it, the model learns the general signal
    "this message contains a link" / "this message contains a long number".
    """
    text = URL_PATTERN.sub(" urltoken ", text.lower())
    return LONG_NUMBER_PATTERN.sub(" longnumber ", text)


def build_models() -> dict:
    """Two text-classification pipelines: TF-IDF -> classifier."""
    def tfidf():
        return TfidfVectorizer(preprocessor=normalise, stop_words="english", ngram_range=(1, 1))

    return {
        "Naive Bayes": make_pipeline(tfidf(), MultinomialNB(alpha=0.1)),
        "Logistic regression": make_pipeline(tfidf(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    }


def evaluate(y_true: pd.Series, y_pred) -> dict:
    """Precision / recall / F1 for the SPAM class (the one we care about)."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=["spam"], average=None, zero_division=0)
    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return {"accuracy": accuracy, "precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])}


def top_words(model, n: int = 8) -> tuple[list[str], list[str]]:
    """Words that push a Naive Bayes model most towards spam / ham."""
    vectorizer, classifier = model.steps[0][1], model.steps[1][1]
    vocab = np.array(vectorizer.get_feature_names_out())
    spam_idx = list(classifier.classes_).index("spam")
    ham_idx = 1 - spam_idx
    ratio = classifier.feature_log_prob_[spam_idx] - classifier.feature_log_prob_[ham_idx]
    order = np.argsort(ratio)
    return list(vocab[order[::-1][:n]]), list(vocab[order[:n]])


# --------------------------------------------------------------------------- #
# 4. Main workflow
# --------------------------------------------------------------------------- #
def run(offline: bool = False) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df, source = load_data(offline)
    print(f"[data] source = {source}")
    print(f"[data] class balance: {df['label'].value_counts().to_dict()}  "
          f"(spam share = {(df['label'] == 'spam').mean():.1%})\n")

    print("Average message statistics by class:")
    stats = message_stats(df)
    print(stats.to_string(), "\n")

    X, y = df["text"], df["label"]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- baselines -----------------------------------------------------------
    always_ham = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=folds)
    baseline = evaluate(y, always_ham)
    keyword_rule = np.where(X.str.contains(r"\bfree\b", case=False, regex=True), "spam", "ham")
    keyword = evaluate(y, keyword_rule)

    # --- real models ---------------------------------------------------------
    results = {"Baseline: always say 'ham'": baseline, "Rule: contains the word 'free'": keyword}
    predictions = {}
    for name, model in build_models().items():
        predictions[name] = cross_val_predict(model, X, y, cv=folds)  # every message predicted by a model that never saw it
        results[name] = evaluate(y, predictions[name])
    table = pd.DataFrame(results).T
    print("5-fold cross-validated results (precision/recall/F1 are for the SPAM class):")
    print(table.round(3).to_string(), "\n")

    # --- what words give spam away? -----------------------------------------
    nb_full = build_models()["Naive Bayes"].fit(X, y)
    spam_words, ham_words = top_words(nb_full)
    print("Top spam words:", ", ".join(spam_words))
    print("Top ham words :", ", ".join(ham_words), "\n")

    # --- try it on new messages ----------------------------------------------
    demo_messages = [
        "Congratulations! You have won a free prize. Call 0900-000-0000 now to claim",
        "Hey, are you free for lunch tomorrow? Let me know",
        "Your account is locked. Verify at http://example-login.test immediately",
        "Can you send me the notes from today's class?",
    ]
    print("Live predictions (Naive Bayes trained on all data):")
    for message, label in zip(demo_messages, nb_full.predict(demo_messages)):
        print(f"  [{label.upper():4}] {message}")
    print()

    # --- plots ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ConfusionMatrixDisplay.from_predictions(y, predictions["Naive Bayes"], labels=["ham", "spam"], ax=axes[0], colorbar=False)
    axes[0].set_title("Naive Bayes - confusion matrix (cross-validated)")
    table[["precision", "recall", "f1"]].plot.bar(ax=axes[1])
    axes[1].set_title("Spam-class metrics by approach")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")
    fig.suptitle(f"Project 04 - spam classifier | data: {source}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spam_results.png"), dpi=120)
    plt.close(fig)

    # --- key insight (computed) ----------------------------------------------
    best_name = table.loc[["Naive Bayes", "Logistic regression"], "f1"].idxmax()
    ham_with_free = int(((X.str.contains(r"\bfree\b", case=False, regex=True)) & (y == "ham")).sum())
    spam_with_free = int(((X.str.contains(r"\bfree\b", case=False, regex=True)) & (y == "spam")).sum())
    length_ratio = stats.loc["spam", "characters"] / stats.loc["ham", "characters"]
    digit_ratio = stats.loc["spam", "digits"] / max(stats.loc["ham", "digits"], 1e-9)
    insight = (
        f"Accuracy is a trap on imbalanced data: always answering 'ham' already scores {baseline['accuracy']:.0%} but catches "
        f"0% of spam. A one-line keyword rule on the word 'free' has precision {keyword['precision']:.0%} but recall of only "
        f"{keyword['recall']:.0%} ({ham_with_free} normal messages contain 'free' too, vs {spam_with_free} spam; "
        f"{1 - keyword['recall']:.0%} of spam never says 'free'). "
        f"{best_name} gets F1={table.loc[best_name, 'f1']:.2f}. Spam messages are {length_ratio:.1f}x longer and contain "
        f"{digit_ratio:.1f}x more digits (phone numbers, prices) than ham."
    )
    print("KEY INSIGHT:", insight)
    print(f"\nPlot saved to: {os.path.join(OUT_DIR, 'spam_results.png')}")
    return {"table": table, "insight": insight, "source": source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 04 - spam vs ham classifier")
    parser.add_argument("--offline", action="store_true", help="use the bundled sample only")
    args = parser.parse_args()
    run(offline=args.offline)


if __name__ == "__main__":
    main()
