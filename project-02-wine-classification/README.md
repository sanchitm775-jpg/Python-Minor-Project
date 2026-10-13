# Project 02 - Wine Cultivar Classification
**Author:** Sanchit Kumar Mishra

### Problem Statement
Identify which of three grape cultivars an Italian wine was made from, using 13 chemical measurements (alcohol, flavanoids, proline, colour intensity, ...).

### Dataset
`sklearn.datasets.load_wine` - 178 wines (59 / 71 / 48 per class), 13 numeric features. It ships **inside scikit-learn**: no download, no internet. (Not a medical or flower dataset.)

### Approach
1. Compare 5 models with 5-fold stratified cross-validation repeated 5 times: kNN on raw features, kNN on scaled features, logistic regression, RBF-SVM, random forest.
2. Investigate *why* scaling matters by looking at each feature's range and variance.
3. Final check on a 25 % hold-out set with the best model (classification report + confusion matrix).
4. Random-forest feature importance, then re-test a model that uses only the top 3 features.

### Result
| Model | CV accuracy |
|---|---|
| SVM (RBF, scaled) | 0.984 |
| Random forest | 0.979 |
| Logistic regression (scaled) | 0.978 |
| kNN k=5, scaled | 0.959 |
| kNN k=5, **raw** | **0.701** |

Hold-out test (best model): accuracy 0.978 on 45 wines. Top features: flavanoids, proline, colour intensity. Logistic regression on just those 3 features: 0.946.

### Key Insight
Without feature scaling kNN scores 70 % but with it 96 %, because the `proline` column alone holds 99.8 % of the total variance and silently drowns out the other 12 features.

### How to Run
```bash
pip install -r requirements.txt
python main.py            # no internet needed - data is built into scikit-learn
python main.py --offline  # force the bundled data (identical numbers to the Result section)
```
Charts and tables are written to the `outputs/` folder next to `main.py`.
