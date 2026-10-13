# Project 04 - Spam vs Ham SMS Classification (NLP basics)
**Author:** Sanchit Kumar Mishra

### Problem Statement
Decide whether a text message is spam or a normal ("ham") message, using only its words.

### Dataset
- **Default:** the UCI *SMS Spam Collection* (5,574 real messages, ~13 % spam), downloaded from the UCI repository, with a GitHub mirror of the same file as a second source.
- **Fallback:** `data/sms_sample.csv` - **164 messages written by me** (53 spam / 111 ham) in the style of the real data, including tricky ham such as "are you free tonight?". It is **not** the UCI data and its spam share (32 %) is higher; it exists so the script runs without internet.

### Approach
1. Look at simple statistics (length, digits, capitals) per class.
2. Normalise text: replace every link with `urltoken` and every long number with `longnumber`, so the model learns "contains a link / phone number" instead of memorising individual URLs.
3. TF-IDF features -> Naive Bayes and Logistic Regression, scored with 5-fold stratified cross-validation (every message is predicted by a model that never saw it).
4. Compare against two baselines: "always ham" and a "contains the word *free*" rule. Report precision/recall/F1 for the **spam** class.

### Result
On the bundled sample (`--offline`):

| Approach | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Always say "ham" | 0.677 | 0.00 | 0.00 | 0.00 |
| Rule: contains "free" | 0.738 | 0.75 | 0.28 | 0.41 |
| Naive Bayes | 0.939 | 0.85 | 0.98 | 0.91 |
| Logistic regression | 0.957 | 0.93 | 0.94 | 0.93 |

Strongest spam words: `longnumber, urltoken, claim, mobile, receive, offer`. *These are numbers for my small hand-written sample, not for the real UCI data - run with internet and the script prints the real-data scores.*

### Key Insight
Accuracy is a trap: always answering "ham" scores 68 % (on this sample) yet catches 0 % of spam, and a "contains *free*" rule has 75 % precision but only 28 % recall because most spam never says "free".

### How to Run
```bash
pip install -r requirements.txt
python main.py            # downloads the UCI data if reachable
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
