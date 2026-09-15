"""
Cleans and scopes the REAL KCC dataset (Kerala 2024, downloaded via the
data.gov.in Data tab -- see README).

Real schema confirmed from the actual file:
    BlockName, Category, CreatedOn, Crop, DistrictName, KCCCallID, KccAns,
    QueryText, QueryType, Season, Sector, StateName, day, month, year

Two corrections vs. the original assumption:
  1. `Category` is a crop-family grouping (Plantation Crops, Fruits, Vegetables...),
     NOT an issue-type category. The real issue-type field is `QueryType`
     (Plant Protection, Nutrient Management, Weather, Field Preparation, ...).
     This matches the proposal's literal hierarchy: Sector -> Category -> QueryType.
  2. ~69% of records are "Government Schemes" / "Training and Exposure Visits" /
     "Crop Insurance" queries -- administrative, not agronomic advisory calls.
     These are filtered out: they're not what the RQs are about (a farmer
     calling back about an unresolved pest/irrigation/fertilizer problem).

Also filters out the ~21% of genuine agronomic answers written in Malayalam
script -- disclosed as a scoping decision (can't hand-verify gold labels on
text you can't read; off-the-shelf English NLI models would perform badly
on it anyway). This still leaves several thousand records.
"""
import re
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

NON_ADVISORY_QUERYTYPES = {"Government Schemes", "Training and Exposure Visits", "Crop Insurance"}
INDIC_SCRIPT_RE = re.compile(r'[ऀ-ॿഀ-ൿ]')  # Devanagari + Malayalam


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ["StateName", "DistrictName", "Crop", "Category", "QueryType",
                "QueryText", "KccAns", "CreatedOn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Available: {list(df.columns)}")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["QueryText", "KccAns", "Crop", "Category", "QueryType", "StateName", "DistrictName"]:
        df[col] = df[col].astype(str).str.strip()

    df = df[df["QueryText"].str.len() > 3]
    df = df[df["KccAns"].str.len() > 3]
    df = df[~df["KccAns"].str.lower().isin(["irrelevant call", "nan", ""])]

    df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
    df = df.dropna(subset=["CreatedOn"])

    df = df.drop_duplicates(subset=["QueryText", "KccAns", "DistrictName", "CreatedOn"])
    return df.reset_index(drop=True)


def filter_to_agronomic_advisory(df: pd.DataFrame) -> pd.DataFrame:
    """Drop administrative query types (govt schemes, training, insurance) --
    not what the advisory-failure RQs are about."""
    before = len(df)
    df = df[~df["QueryType"].isin(NON_ADVISORY_QUERYTYPES)]
    print(f"Filtered to agronomic-advisory QueryTypes: {before} -> {len(df)} "
          f"({len(df)/before*100:.1f}% kept)")
    return df.reset_index(drop=True)


def filter_to_english_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records whose KccAns contains real Devanagari/Malayalam script.
    Scoping decision, documented in the README/report: keeps the dataset
    hand-labelable and usable with English-only NLI/embedding models."""
    before = len(df)
    is_indic = df["KccAns"].apply(lambda t: bool(INDIC_SCRIPT_RE.search(t)))
    df = df[~is_indic]
    print(f"Filtered to English-script answers: {before} -> {len(df)} "
          f"({len(df)/before*100:.1f}% kept)")
    return df.reset_index(drop=True)


def scope(df: pd.DataFrame, states=None, crops=None, districts=None,
          start_date=None, end_date=None) -> pd.DataFrame:
    out = df.copy()
    if states:
        out = out[out["StateName"].isin(states)]
    if crops:
        out = out[out["Crop"].isin(crops)]
    if districts:
        out = out[out["DistrictName"].isin(districts)]
    if start_date:
        out = out[out["CreatedOn"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["CreatedOn"] <= pd.to_datetime(end_date)]
    return out.reset_index(drop=True)


def _tfidf_cluster_labels(labels: list, distance_threshold=0.3):
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 1:
        return {l: l for l in unique_labels}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform(unique_labels)
    sim = cosine_similarity(X)
    dist = 1 - sim
    dist[dist < 0] = 0
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        metric="precomputed", linkage="average",
    )
    cluster_ids = clustering.fit_predict(dist)
    from collections import Counter
    counts = Counter(labels)
    cluster_to_labels = {}
    for lbl, cid in zip(unique_labels, cluster_ids):
        cluster_to_labels.setdefault(cid, []).append(lbl)
    mapping = {}
    for cid, lbls in cluster_to_labels.items():
        canonical = max(lbls, key=lambda l: counts[l])
        for l in lbls:
            mapping[l] = canonical
    return mapping


def merge_near_duplicate_labels(df: pd.DataFrame, column="QueryType", distance_threshold=0.3) -> pd.DataFrame:
    df = df.copy()
    mapping = _tfidf_cluster_labels(df[column].tolist(), distance_threshold=distance_threshold)
    df[f"{column}_raw"] = df[column]
    df[column] = df[column].map(mapping)
    print(f"Merged {column}: {df[f'{column}_raw'].nunique()} raw labels -> {df[column].nunique()} canonical")
    return df


def run(raw_path, out_path, states=None, crops=None, districts=None,
        start_date=None, end_date=None, keep_non_english=False, keep_all_querytypes=False):
    df = load_raw(raw_path)
    print(f"Loaded {len(df)} raw records")
    df = basic_clean(df)
    print(f"After basic cleaning: {len(df)} records")
    if not keep_all_querytypes:
        df = filter_to_agronomic_advisory(df)
    if not keep_non_english:
        df = filter_to_english_answers(df)
    df = scope(df, states=states, crops=crops, districts=districts,
               start_date=start_date, end_date=end_date)
    print(f"After scoping: {len(df)} records")
    df = merge_near_duplicate_labels(df, column="QueryType")
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned+scoped data -> {out_path}")
    print("\nFinal QueryType distribution:")
    print(df["QueryType"].value_counts())
    print("\nFinal Crop distribution (top 10):")
    print(df["Crop"].value_counts().head(10))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/kcc_kerala_2024.csv")
    parser.add_argument("--out", default="data/processed/kcc_clean.csv")
    parser.add_argument("--states", nargs="*", default=None)
    parser.add_argument("--crops", nargs="*", default=None)
    parser.add_argument("--districts", nargs="*", default=None)
    parser.add_argument("--keep-non-english", action="store_true")
    parser.add_argument("--keep-all-querytypes", action="store_true")
    args = parser.parse_args()
    run(args.raw, args.out, states=args.states, crops=args.crops, districts=args.districts,
        keep_non_english=args.keep_non_english, keep_all_querytypes=args.keep_all_querytypes)
