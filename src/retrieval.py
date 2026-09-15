"""
Retrieval component (secondary/stretch, per proposal): given a new query,
surface similar past resolved query-answer pairs.

Same two-backend pattern as semantic_mismatch.py:
  --backend tfidf   TF-IDF + cosine nearest-neighbours. No external
                     downloads, works anywhere. Weaker semantically (misses
                     paraphrases) but fine as a first pass / fallback.
  --backend sbert    sentence-transformers embeddings + FAISS, as proposed.
                     Needs HuggingFace access -- run on your machine.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")
        self.matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame, text_col="QueryText"):
        self.df = df.reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(self.df[text_col])
        return self

    def query(self, text: str, k=5):
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results[["QueryText", "KccAns", "Crop", "Category", "similarity"]]


class SbertFaissRetriever:
    """As proposed: sentence-transformers embeddings + FAISS index.
    Requires `sentence-transformers`, `faiss-cpu`, and HuggingFace access
    (blocked in the dev sandbox this was written in) -- run on your machine."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss
        self.model = SentenceTransformer(model_name)
        self.faiss = faiss
        self.index = None
        self.df = None

    def fit(self, df: pd.DataFrame, text_col="QueryText"):
        self.df = df.reset_index(drop=True)
        embeddings = self.model.encode(self.df[text_col].tolist(), normalize_embeddings=True)
        self.index = self.faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype("float32"))
        return self

    def query(self, text: str, k=5):
        emb = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(emb, k)
        results = self.df.iloc[idx[0]].copy()
        results["similarity"] = scores[0]
        return results[["QueryText", "KccAns", "Crop", "Category", "similarity"]]


def build_retriever(df: pd.DataFrame, backend="tfidf"):
    if backend == "tfidf":
        return TfidfRetriever().fit(df)
    elif backend == "sbert":
        return SbertFaissRetriever().fit(df)
    raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/kcc_clean.csv")
    parser.add_argument("--backend", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    retriever = build_retriever(df, backend=args.backend)
    results = retriever.query(args.query, k=args.k)
    print(results.to_string(index=False))
