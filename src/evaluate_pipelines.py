"""
RQ1: does LLM-based structured extraction beat the supervised TF-IDF+SVM
baseline classifier at predicting QueryType (the real 42-value issue-type
taxonomy), when both are scored against the same hand-verified gold set?

Run this AFTER:
    python src/baseline_classifier.py
    python src/llm_extraction.py --input data/gold/gold_set_final.csv \
        --out outputs/llm_extraction_gold.csv --model qwen2.5:7b-instruct
"""
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import baseline_classifier as bc


def load_and_align(gold_path, llm_path):
    gold = pd.read_csv(gold_path)
    llm = pd.read_csv(llm_path)

    if len(gold) != len(llm):
        print(f"WARNING: gold set has {len(gold)} rows but LLM extraction has {len(llm)} rows -- "
              "did you run llm_extraction.py on data/gold/gold_set_final.csv specifically?")

    merged = gold.merge(llm, on="KCCCallID", suffixes=("_gold", "_llm"), how="inner")
    n_dropped = len(gold) - len(merged)
    if n_dropped:
        print(f"WARNING: {n_dropped} gold rows had no matching KCCCallID in the LLM extraction output "
              "and were dropped from this comparison.")
    if len(merged) == 0:
        raise SystemExit("No overlapping KCCCallID between the gold set and the LLM extraction output.")
    return gold, llm, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold/gold_set_final.csv")
    parser.add_argument("--llm", default="outputs/llm_extraction_gold.csv")
    parser.add_argument("--model-path", default="models/baseline_classifier.joblib")
    parser.add_argument("--out", default="outputs/rq1_comparison.json")
    args = parser.parse_args()

    gold, llm, merged = load_and_align(args.gold, args.llm)

    required_cols = ["QueryType", "resolution_signal_gold", "resolution_signal_llm", "category"]
    for col in required_cols:
        n_missing = merged[col].isna().sum()
        if n_missing:
            raise SystemExit(f"{n_missing} rows have a blank '{col}' value -- fill in every row before scoring.")

    llm_issuetype_acc = accuracy_score(merged["QueryType"], merged["category"])
    llm_issuetype_f1 = f1_score(merged["QueryType"], merged["category"], average="macro", zero_division=0)
    llm_resolution_acc = accuracy_score(merged["resolution_signal_gold"], merged["resolution_signal_llm"])

    clf = bc.HierarchicalClassifier.load(args.model_path)
    _, pred_qtypes = clf.predict(merged["QueryText"].tolist())
    base_issuetype_acc = accuracy_score(merged["QueryType"], pred_qtypes)
    base_issuetype_f1 = f1_score(merged["QueryType"], pred_qtypes, average="macro", zero_division=0)

    winner = "LLM extraction" if llm_issuetype_acc > base_issuetype_acc else "Baseline classifier"
    if llm_issuetype_acc == base_issuetype_acc:
        winner = "Tie"

    results = {
        "n_compared": len(merged),
        "llm_issuetype_accuracy": llm_issuetype_acc,
        "llm_issuetype_f1_macro": llm_issuetype_f1,
        "llm_resolution_signal_accuracy": llm_resolution_acc,
        "baseline_issuetype_accuracy": base_issuetype_acc,
        "baseline_issuetype_f1_macro": base_issuetype_f1,
        "winner": winner,
    }

    print("=== RQ1: LLM extraction vs supervised baseline (QueryType accuracy on gold set) ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(f"\nRQ1 answer: {winner} scores higher QueryType accuracy on the hand-verified gold set "
          f"({llm_issuetype_acc:.3f} vs {base_issuetype_acc:.3f}, n={len(merged)}).")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()