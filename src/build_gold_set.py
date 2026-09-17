"""
Samples a stratified subset of cleaned records for hand-labeling, and
provides a merge step to bring your corrections back in.

Step 1: python src/build_gold_set.py sample --n 100
        -> writes data/gold/gold_set_to_label.csv with the existing
           (noisy) Category/QueryType pre-filled as a starting point,
           plus empty *_verified columns.

Step 2: open that CSV (Excel/Google Sheets/VS Code), and for each row either
        confirm the pre-filled label is right or correct it in the
        *_verified column. Faster than labeling from scratch since most
        rows the agent already got roughly right.

Step 3: python src/build_gold_set.py merge
        -> writes data/gold/gold_set_final.csv with verified labels only,
           ready to be used as ground truth in baseline_classifier.py's
           evaluate_against_gold() and in the RQ1 comparison against the
           LLM extraction output.

           This is pretty annoying if you don't have good computational power
"""
import argparse
import pandas as pd


def sample_gold(input_path, out_path, n=100, seed=42):
    df = pd.read_csv(input_path)
    # stratify roughly by Category so rare categories aren't left out
    n_per_cat = max(1, n // df["Category"].nunique())
    parts = [group.sample(min(len(group), n_per_cat), random_state=seed)
             for _, group in df.groupby("Category")]
    sampled = pd.concat(parts)
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(min(len(remaining), n - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    sampled = sampled.reset_index(drop=True)
    sampled["Category_verified"] = sampled["Category"]   # pre-filled starting point
    sampled["QueryType_verified"] = sampled["QueryType"]  # pre-filled starting point
    sampled["resolution_signal_verified"] = ""  # you fill this by hand: resolved / unresolved / unclear
    sampled["notes"] = ""

    id_col = "KCCCallID" if "KCCCallID" in sampled.columns else "RecordId"
    cols = [id_col, "StateName", "DistrictName", "Crop", "Season", "QueryText", "KccAns",
            "Category", "QueryType", "Category_verified", "QueryType_verified",
            "resolution_signal_verified", "notes"]
    sampled[cols].to_csv(out_path, index=False)
    print(f"Sampled {len(sampled)} records -> {out_path}")
    print("Open this file, correct the *_verified columns where the pre-filled label is wrong, "
          "and fill in resolution_signal_verified for each row.")


def merge_gold(labeled_path, out_path):
    df = pd.read_csv(labeled_path)
    missing_verif = df["resolution_signal_verified"].isna().sum() + (df["resolution_signal_verified"] == "").sum()
    if missing_verif:
        print(f"WARNING: {missing_verif} rows still have an empty resolution_signal_verified. "
              "Fill those in before using this as ground truth.")
    final = df.rename(columns={
        "Category_verified": "Category",
        "QueryType_verified": "QueryType",
        "resolution_signal_verified": "resolution_signal",
    })
    final = final.drop(columns=[c for c in ["notes"] if c in final.columns])
    final.to_csv(out_path, index=False)
    print(f"Saved final gold set ({len(final)} records) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--input", default="data/processed/kcc_clean.csv")
    p_sample.add_argument("--out", default="data/gold/gold_set_to_label.csv")
    p_sample.add_argument("--n", type=int, default=100)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--input", default="data/gold/gold_set_to_label.csv")
    p_merge.add_argument("--out", default="data/gold/gold_set_final.csv")

    args = parser.parse_args()
    if args.cmd == "sample":
        sample_gold(args.input, args.out, n=args.n)
    else:
        merge_gold(args.input, args.out)
