"""
Layer 2: Build a PM4Py event log from the validated Layer 1 output and mine it.

Per proposal: case = district + crop, activity = validated category,
timestamp = call date. "Validated category" means whichever of the baseline
classifier / LLM extraction pipeline scored better against the gold set in
Layer 1 (decided in evaluate_pipelines.py) -- here it's a parameter
(`activity_col`) so this module doesn't care which one produced it.

Outputs:
  - a PM4Py-formatted event log
  - a discovered process model (Inductive Miner -> guaranteed sound/readable)
  - loop rate per case (proxy for "unresolved query": same activity recurring)
  - case duration
  - variant analysis, sliceable by crop / state / season
"""
import argparse
import pandas as pd
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.conversion.process_tree import converter as pt_converter


def build_event_log_df(df: pd.DataFrame, activity_col="Category", date_col="CreatedOn") -> pd.DataFrame:
    log_df = df.copy()
    log_df["case:concept:name"] = log_df["DistrictName"].astype(str) + "_" + log_df["Crop"].astype(str)
    log_df["concept:name"] = log_df[activity_col].astype(str)
    log_df["time:timestamp"] = pd.to_datetime(log_df[date_col])
    log_df = log_df.sort_values(["case:concept:name", "time:timestamp"])
    return log_df[["case:concept:name", "concept:name", "time:timestamp",
                    "StateName", "Crop", "Season", "QueryText", "KccAns"]]


def discover_model(log_df: pd.DataFrame, top_n_activities=8, top_k_variants=15):
    """Discovers a process model for VISUALIZATION. With 40+ distinct
    QueryTypes and 700+ real-world-variable cases, Inductive Miner is still
    "sound" (a valid, deadlock-free Petri net) but visually turns into an
    unreadable spaghetti model -- this is a real limitation worth disclosing,
    not a bug. Standard fix (what real process mining tools do): filter to
    the most frequent activities and variants before discovery, so the MAP
    is legible, while loop-rate/duration/variant CSVs keep full resolution.
    """
    filtered = log_df.copy()

    top_activities = filtered["concept:name"].value_counts().head(top_n_activities).index
    filtered["concept:name"] = filtered["concept:name"].where(
        filtered["concept:name"].isin(top_activities), "Other")

    event_log = pm4py.convert_to_event_log(filtered)
    if top_k_variants:
        event_log = pm4py.filter_variants_top_k(event_log, top_k_variants)

    tree = inductive_miner.apply(event_log)
    net, im, fm = pt_converter.apply(tree)
    return event_log, tree, net, im, fm


def export_process_map(net, im, fm, out_path="outputs/process_model.png"):
    pm4py.save_vis_petri_net(net, im, fm, out_path)
    print(f"Saved process map -> {out_path}")


def discover_and_export_dfg(log_df: pd.DataFrame, out_path="outputs/process_map.png",
                             top_n_activities=6, top_n_edges=10):
    """A directly-follows graph (DFG) instead of a raw Petri net: plain
    labeled boxes + frequency-weighted arrows, no places/silent-transition
    notation. This is what tools like Celonis/Disco show as "the process
    map" -- much more readable for a demo/report than the Inductive Miner
    Petri net, which is technically correct but dense academic notation.

    Trims to top_n_activities nodes and top_n_edges edges BY HAND (rather
    than relying on pm4py's own max_num_edges option, which has a bug where
    it can drop a node's only edges while leaving it listed as a start/end
    activity, causing a KeyError deep in the graphviz renderer).
    """
    filtered = log_df.copy()
    top_activities = filtered["concept:name"].value_counts().head(top_n_activities).index
    filtered["concept:name"] = filtered["concept:name"].where(
        filtered["concept:name"].isin(top_activities), "Other")
    # sanitize labels for the graph renderer only (commas inside a label
    # like "Water Management, Micro Irrigation" break pm4py/graphviz) --
    # this only affects the picture, not the underlying data/CSVs
    filtered["concept:name"] = filtered["concept:name"].str.replace(",", " -", regex=False)

    dfg, start_activities, end_activities = pm4py.discover_dfg(
        filtered, activity_key="concept:name", timestamp_key="time:timestamp",
        case_id_key="case:concept:name")

    # keep only the strongest edges, and drop self-loops from the edge-count
    # ranking so a couple of huge repeat-loops don't crowd out everything else
    non_loop = {e: c for e, c in dfg.items() if e[0] != e[1]}
    loop = {e: c for e, c in dfg.items() if e[0] == e[1]}
    top_non_loop = dict(sorted(non_loop.items(), key=lambda kv: kv[1], reverse=True)[:top_n_edges])
    trimmed_dfg = {**top_non_loop, **loop}

    nodes_in_graph = set()
    for (a, b) in trimmed_dfg:
        nodes_in_graph.add(a)
        nodes_in_graph.add(b)
    start_activities = {a: c for a, c in start_activities.items() if a in nodes_in_graph}
    end_activities = {a: c for a, c in end_activities.items() if a in nodes_in_graph}

    pm4py.save_vis_dfg(trimmed_dfg, start_activities, end_activities, out_path,
                        rankdir="LR",
                        graph_title="KCC Advisory Process Map (strongest flows, by call frequency)")
    print(f"Saved directly-follows process map -> {out_path}")


def discover_dfg_data(log_df: pd.DataFrame, top_n_activities=6, top_n_edges=10):
    """Same discovery + trimming logic as discover_and_export_dfg, but returns
    the raw (trimmed_dfg, start_activities, end_activities, node_totals) data
    instead of rendering a graphviz PNG -- so the dashboard can draw its own
    interactive Plotly version (hoverable counts, no static image) from the
    exact same numbers that appear in the static report figure.
    """
    filtered = log_df.copy()
    top_activities = filtered["concept:name"].value_counts().head(top_n_activities).index
    filtered["concept:name"] = filtered["concept:name"].where(
        filtered["concept:name"].isin(top_activities), "Other")
    filtered["concept:name"] = filtered["concept:name"].str.replace(",", " -", regex=False)

    dfg, start_activities, end_activities = pm4py.discover_dfg(
        filtered, activity_key="concept:name", timestamp_key="time:timestamp",
        case_id_key="case:concept:name")

    non_loop = {e: c for e, c in dfg.items() if e[0] != e[1]}
    loop = {e: c for e, c in dfg.items() if e[0] == e[1]}
    top_non_loop = dict(sorted(non_loop.items(), key=lambda kv: kv[1], reverse=True)[:top_n_edges])
    trimmed_dfg = {**top_non_loop, **loop}

    nodes_in_graph = set()
    for (a, b) in trimmed_dfg:
        nodes_in_graph.add(a)
        nodes_in_graph.add(b)
    start_activities = {a: c for a, c in start_activities.items() if a in nodes_in_graph}
    end_activities = {a: c for a, c in end_activities.items() if a in nodes_in_graph}

    # total call volume per node, straight from the (pre-trim) activity counts,
    # so node size/labels reflect true frequency even if some of a node's
    # edges got trimmed out of the picture for readability
    node_totals = filtered["concept:name"].value_counts().to_dict()
    node_totals = {n: node_totals.get(n, 0) for n in nodes_in_graph}

    return trimmed_dfg, start_activities, end_activities, node_totals


def loop_rate_per_case(log_df: pd.DataFrame) -> pd.DataFrame:
    """Loop rate = fraction of events in a case whose activity repeats an
    earlier activity in the same case. High loop rate = proxy for
    "farmer kept calling back about the same category of issue"."""
    rows = []
    for case_id, group in log_df.groupby("case:concept:name"):
        activities = group["concept:name"].tolist()
        seen = set()
        repeats = 0
        for a in activities:
            if a in seen:
                repeats += 1
            seen.add(a)
        loop_rate = repeats / len(activities) if activities else 0
        rows.append({
            "case_id": case_id,
            "n_events": len(activities),
            "n_repeats": repeats,
            "loop_rate": loop_rate,
            "crop": group["Crop"].iloc[0],
            "state": group["StateName"].iloc[0],
        })
    return pd.DataFrame(rows)


def case_durations(log_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in log_df.groupby("case:concept:name"):
        duration_days = (group["time:timestamp"].max() - group["time:timestamp"].min()).days
        rows.append({
            "case_id": case_id,
            "duration_days": duration_days,
            "n_events": len(group),
            "crop": group["Crop"].iloc[0],
            "state": group["StateName"].iloc[0],
        })
    return pd.DataFrame(rows)


def variant_analysis(log_df: pd.DataFrame) -> pd.DataFrame:
    """Per-case activity sequence ('variant'), so we can see the most common
    paths and break them down by crop/state/season."""
    rows = []
    for case_id, group in log_df.groupby("case:concept:name"):
        variant = " -> ".join(group["concept:name"].tolist())
        rows.append({
            "case_id": case_id,
            "variant": variant,
            "crop": group["Crop"].iloc[0],
            "state": group["StateName"].iloc[0],
            "season": group["Season"].mode().iloc[0] if len(group["Season"].mode()) else None,
        })
    return pd.DataFrame(rows)


def summarize_by_crop(loop_df: pd.DataFrame, duration_df: pd.DataFrame) -> pd.DataFrame:
    merged = loop_df.merge(duration_df[["case_id", "duration_days"]], on="case_id")
    summary = merged.groupby("crop").agg(
        mean_loop_rate=("loop_rate", "mean"),
        mean_duration_days=("duration_days", "mean"),
        n_cases=("case_id", "count"),
    ).reset_index().sort_values("mean_loop_rate", ascending=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/kcc_clean.csv")
    parser.add_argument("--activity-col", default="QueryType")  # real issue-type field
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--drop-others-crop", action="store_true", default=True,
                         help="Drop Crop=='Others' rows -- case ID (district+crop) needs a real crop to be meaningful")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.drop_others_crop:
        before = len(df)
        df = df[df["Crop"] != "Others"].reset_index(drop=True)
        print(f"Dropped Crop=='Others' rows for event log (case ID needs a real crop): {before} -> {len(df)}")
    log_df = build_event_log_df(df, activity_col=args.activity_col)
    print(f"Built event log: {log_df['case:concept:name'].nunique()} cases, {len(log_df)} events")

    event_log, tree, net, im, fm = discover_model(log_df)
    export_process_map(net, im, fm, out_path=f"{args.out_dir}/process_model.png")
    discover_and_export_dfg(log_df, out_path=f"{args.out_dir}/process_map.png")

    loop_df = loop_rate_per_case(log_df)
    dur_df = case_durations(log_df)
    var_df = variant_analysis(log_df)
    summary = summarize_by_crop(loop_df, dur_df)

    loop_df.to_csv(f"{args.out_dir}/loop_rate_per_case.csv", index=False)
    dur_df.to_csv(f"{args.out_dir}/case_durations.csv", index=False)
    var_df.to_csv(f"{args.out_dir}/variants.csv", index=False)
    summary.to_csv(f"{args.out_dir}/summary_by_crop.csv", index=False)

    print("\n=== Loop rate & duration by crop ===")
    print(summary.to_string(index=False))
    print(f"\nTop 5 variants:\n{var_df['variant'].value_counts().head()}")
