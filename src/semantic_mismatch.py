"""
RQ3: Does the answer actually address the query, or is it a generic/deflecting
non-answer that conventional process metrics (call timing, repeat-call count)
wouldn't catch on their own?

Two backends, same interface:

  --backend nli      (as proposed): a pretrained NLI model scores whether the
                      answer entails "this response addresses: <query>".
                      Needs HuggingFace access -- run this on your machine,
                      not in a network-restricted sandbox.

  --backend lexical   fallback: TF-IDF cosine similarity between query and
                      answer as a cruder proxy for "does the answer even
                      engage with the same topic as the query". No external
                      downloads, works anywhere -- useful for testing the
                      pipeline plumbing before the NLI backend is available,
                      and as a documented fallback if HF access falls through.

Both output the same columns: mismatch_score (0=clearly addresses it,
1=clearly doesn't), is_mismatch (bool, thresholded).
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def lexical_mismatch_scores(df: pd.DataFrame, query_col="QueryText", answer_col="KccAns") -> np.ndarray:
    """1 - cosine_similarity(query, answer) under shared TF-IDF space.
    Crude but dependency-free proxy: an answer sharing almost no vocabulary
    with the query (e.g. a generic deflection) scores as a likely mismatch."""
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    corpus = pd.concat([df[query_col], df[answer_col]])
    vec.fit(corpus)
    Xq = vec.transform(df[query_col])
    Xa = vec.transform(df[answer_col])
    sims = np.array([cosine_similarity(Xq[i], Xa[i])[0, 0] for i in range(len(df))])
    return 1 - sims


def nli_mismatch_scores(df: pd.DataFrame, query_col="QueryText", answer_col="KccAns",
                         model_name="cross-encoder/nli-deberta-v3-xsmall", batch_size=16) -> np.ndarray:
    """Real NLI-based scoring as proposed. Requires `transformers` + `torch`
    and HuggingFace Hub access to download model weights (blocked in the
    dev sandbox this was written in -- run on a machine with normal internet).

    Premise = the answer. Hypothesis = "This response addresses the
    farmer's concern: {query}". Mismatch score = 1 - P(entailment).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # find which output index corresponds to "entailment" for this model
    id2label = {v.lower(): k for k, v in model.config.id2label.items()}
    entail_idx = id2label.get("entailment", 0)

    scores = []
    premises = df[answer_col].tolist()
    hypotheses = [f"This response addresses the farmer's concern: {q}" for q in df[query_col]]

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_p = premises[i:i + batch_size]
            batch_h = hypotheses[i:i + batch_size]
            inputs = tokenizer(batch_p, batch_h, return_tensors="pt", padding=True, truncation=True)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            entail_probs = probs[:, entail_idx].tolist()
            scores.extend([1 - p for p in entail_probs])
    return np.array(scores)


def run(df: pd.DataFrame, backend="lexical", threshold=0.5, **kwargs) -> pd.DataFrame:
    out = df.copy()
    if backend == "lexical":
        scores = lexical_mismatch_scores(df, **kwargs)
    elif backend == "nli":
        scores = nli_mismatch_scores(df, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    out["mismatch_score"] = scores
    out["is_mismatch"] = out["mismatch_score"] >= threshold
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/kcc_clean.csv")
    parser.add_argument("--backend", choices=["lexical", "nli"], default="lexical")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/semantic_mismatch.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = run(df, backend=args.backend, threshold=args.threshold)
    result.to_csv(args.out, index=False)
    print(f"Flagged {result['is_mismatch'].sum()}/{len(result)} records as likely mismatches "
          f"(backend={args.backend})")
    print(result[["QueryText", "KccAns", "mismatch_score", "is_mismatch"]].sort_values(
        "mismatch_score", ascending=False).head(5).to_string())
