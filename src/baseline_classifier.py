"""
Layer 1, path A: Hierarchical TF-IDF + Linear SVM baseline classifier.

Predicts Category first (from QueryText), then QueryType conditioned on the
predicted Category (a separate classifier per category, trained only on that
category's records) -- this preserves the parent/child structure instead of
flattening into one big multiclass problem.

Trained on the noisy Category/QueryType labels already in the data (weak
ground truth). Evaluated separately against the small hand-labeled gold set
in evaluate_against_gold().
"""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


class HierarchicalClassifier:
    def __init__(self, min_class_count=3, C=2.0):
        # sublinear_tf (log-scaled term frequency, standard for TF-IDF+linear-SVM
        # text classification) and min_df=2 (drop terms that appear exactly once
        # anywhere -- almost always typos/noise, not learnable signal) are the two
        # vectorizer changes from the original config, both generic best practice.
        # C is tuned via a validation split carved out of the TRAINING data only
        # (see tune_C.py) -- the gold set is never touched during tuning.
        self.category_vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2),
                                                     min_df=2, sublinear_tf=True)
        self.category_clf = LinearSVC(class_weight="balanced", C=C)
        self.querytype_vectorizers = {}   # category -> vectorizer
        self.querytype_clfs = {}          # category -> classifier
        self.min_class_count = min_class_count
        self.C = C

    def fit(self, df: pd.DataFrame, text_col="QueryText", category_col="Category", querytype_col="QueryType"):
        X_text = df[text_col].values
        y_cat = df[category_col].values

        Xc = self.category_vectorizer.fit_transform(X_text)
        self.category_clf.fit(Xc, y_cat)

        for cat in df[category_col].unique():
            sub = df[df[category_col] == cat]
            vc = sub[querytype_col].value_counts()
            # drop QueryType classes with too few examples to be learnable
            keep_classes = vc[vc >= self.min_class_count].index
            sub = sub[sub[querytype_col].isin(keep_classes)]
            if sub[querytype_col].nunique() < 2 or len(sub) < 4:
                # not enough signal to train a sub-classifier; fall back to majority class
                self.querytype_vectorizers[cat] = None
                self.querytype_clfs[cat] = sub[querytype_col].mode().iloc[0] if len(sub) else None
                continue
            # min_df stays at 1 here (unlike the category vectorizer above) --
            # some categories have only a few dozen records, and min_df=2 on
            # a subset that small starves the vocabulary rather than denoising it
            vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            Xq = vec.fit_transform(sub[text_col])
            clf = LinearSVC(class_weight="balanced", C=self.C)
            clf.fit(Xq, sub[querytype_col])
            self.querytype_vectorizers[cat] = vec
            self.querytype_clfs[cat] = clf
        return self

    def predict(self, texts):
        Xc = self.category_vectorizer.transform(texts)
        pred_cats = self.category_clf.predict(Xc)

        pred_qtypes = []
        for text, cat in zip(texts, pred_cats):
            vec = self.querytype_vectorizers.get(cat)
            clf = self.querytype_clfs.get(cat)
            if vec is None:
                # fallback: majority class stored directly as `clf`
                pred_qtypes.append(clf if isinstance(clf, str) else "UNKNOWN")
            else:
                Xq = vec.transform([text])
                pred_qtypes.append(clf.predict(Xq)[0])
        return pred_cats, np.array(pred_qtypes)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def evaluate_against_gold(model: HierarchicalClassifier, gold_df: pd.DataFrame,
                           text_col="QueryText", category_col="Category", querytype_col="QueryType"):
    pred_cats, pred_qtypes = model.predict(gold_df[text_col].tolist())
    true_cats = gold_df[category_col].values
    true_qtypes = gold_df[querytype_col].values

    results = {
        "category_accuracy": accuracy_score(true_cats, pred_cats),
        "category_f1_macro": f1_score(true_cats, pred_cats, average="macro", zero_division=0),
        "querytype_accuracy": accuracy_score(true_qtypes, pred_qtypes),
        "querytype_f1_macro": f1_score(true_qtypes, pred_qtypes, average="macro", zero_division=0),
    }
    print("=== Baseline classifier vs gold set ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
    print("\nCategory report:\n", classification_report(true_cats, pred_cats, zero_division=0))
    return results


if __name__ == "__main__":
    # Re-import this file as a normal module (not "__main__") before training,
    # so the pickled class's __module__ is "baseline_classifier" -- otherwise
    # joblib.load() fails with "Can't get attribute 'HierarchicalClassifier'
    # on <module '__main__'>" when loaded from anywhere else (e.g. the dashboard).
    import baseline_classifier as _bc

    df = pd.read_csv("data/processed/kcc_clean.csv")

    # merge Category classes with too few members to stratify/learn from (e.g. a
    # literal "0" placeholder, or a category with a single record) into "Rare/Other"
    cat_counts = df["Category"].value_counts()
    rare_cats = cat_counts[cat_counts < 5].index
    if len(rare_cats):
        print(f"Merging {len(rare_cats)} rare Category values into 'Rare/Other': {list(rare_cats)}")
        df["Category"] = df["Category"].where(~df["Category"].isin(rare_cats), "Rare/Other")

    train_df, gold_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Category"])
    print(f"Train: {len(train_df)}, held-out (stand-in for gold set): {len(gold_df)}")

    model = _bc.HierarchicalClassifier()
    model.fit(train_df)
    model.save("models/baseline_classifier.joblib")

    metrics = evaluate_against_gold(model, gold_df)
    with open("outputs/baseline_classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)