"""
Layer 3: Dashboard. Reads the outputs produced by baseline_classifier.py,
event_log.py, evaluate_pipelines.py and semantic_mismatch.py, and ties them
into a findings summary + KPI tiles + process map + loop-rate chart + an
interactive per-case event explorer + a live query demo.

Run: python dashboard/app.py   (then open http://127.0.0.1:8050)
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from baseline_classifier import HierarchicalClassifier  # noqa: E402,F401 -- looks unused, but joblib.load()
# needs this exact class importable under this exact name to reconstruct the pickled classifier below --
# deleting this import will break loading models/baseline_classifier.joblib
from event_log import (build_event_log_df, discover_model,  # noqa: E402
                        discover_dfg_data, extract_petri_layout)
from retrieval import TfidfRetriever  # noqa: E402 -- sklearn only, no torch/HF download needed

BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "outputs")

loop_df = pd.read_csv(os.path.join(OUT, "loop_rate_per_case.csv"))
dur_df = pd.read_csv(os.path.join(OUT, "case_durations.csv"))
var_df = pd.read_csv(os.path.join(OUT, "variants.csv"))
summary_df = pd.read_csv(os.path.join(OUT, "summary_by_crop.csv"))
mismatch_df = pd.read_csv(os.path.join(OUT, "semantic_mismatch.csv")) if os.path.exists(
    os.path.join(OUT, "semantic_mismatch.csv")) else None
llm_df = pd.read_csv(os.path.join(OUT, "llm_extraction.csv")) if os.path.exists(
    os.path.join(OUT, "llm_extraction.csv")) else None

rq1_path = os.path.join(OUT, "rq1_comparison.json")
rq1 = json.load(open(rq1_path)) if os.path.exists(rq1_path) else None

classifier_path = os.path.join(BASE, "models", "baseline_classifier.joblib")
classifier = joblib.load(classifier_path) if os.path.exists(classifier_path) else None

# --- raw per-call data, kept ONLY for the case explorer (event-level drilldown) ---
# case = DistrictName + "_" + Crop, activity = QueryType, ordered by CreatedOn --
# same definition event_log.py uses (and the same field the static report
# figures/CSVs were built from), so the explorer matches the loop-rate/duration
# numbers exactly. NOTE: this used to be "Category", which is actually the
# crop's broad family (Vegetables/Fruits/Plantation Crops/...) and barely
# varies within a case defined by district+crop -- QueryType (the real
# 42-value advisory issue-type field) is what actually changes call to call
# and is what the process map is supposed to show.
ACTIVITY_COL = "QueryType"
raw_df = pd.read_csv(os.path.join(BASE, "data", "processed", "kcc_clean.csv"))
raw_df["case_id"] = raw_df["DistrictName"].astype(str) + "_" + raw_df["Crop"].astype(str)
raw_df["CreatedOn"] = pd.to_datetime(raw_df["CreatedOn"])
raw_df = raw_df.sort_values(["case_id", "CreatedOn"])

# TF-IDF retriever for the live demo -- surfaces REAL past KCC answers for
# similar questions instead of the classifier just naming a category and
# stopping. Deliberately the tfidf backend, not sbert: no torch/HuggingFace
# download needed, so this works the same locally and on a deployed instance.
retriever = TfidfRetriever().fit(raw_df, text_col="QueryText")

CROPS = sorted(loop_df["crop"].unique())
# every case in this dataset is KERALA -- see the "Honest deviations" section of
# the README for why this project is scoped to one state rather than filtered
# down from a larger multi-state pull. No state filter is shown because a
# dropdown with one possible value is dead UI, not a real filter.

# case list for the explorer dropdown -- busiest / most-looping cases first, since
# those are the interesting ones to drill into
case_options_df = (loop_df.merge(dur_df[["case_id", "duration_days"]], on="case_id", how="left")
                    .sort_values("n_events", ascending=False))
CASE_OPTIONS = [
    {"label": f"{r.case_id}  —  {r.n_events} events, loop rate {r.loop_rate:.0%}", "value": r.case_id}
    for r in case_options_df.itertuples()
]
DEFAULT_CASE = case_options_df.iloc[0]["case_id"] if len(case_options_df) else None

# variant explorer: how many CASES share the exact same activity sequence --
# a different question from the case explorer (which is "what happened in
# THIS one case"). Distinct from case-level detail: this groups cases by
# shared pattern so recurring advisory pathways are visible at a glance.
variant_counts = (var_df.groupby("variant")
                  .agg(n_cases=("case_id", "nunique"),
                       example_crops=("crop", lambda s: ", ".join(sorted(set(s))[:3])))
                  .reset_index())
variant_counts["n_steps"] = variant_counts["variant"].str.count("->") + 1
variant_counts["pct_of_cases"] = (variant_counts["n_cases"] / var_df["case_id"].nunique() * 100).round(1)
variant_counts = variant_counts.sort_values("n_cases", ascending=False).reset_index(drop=True)
variant_counts.insert(0, "rank", range(1, len(variant_counts) + 1))

# --- derived, report-ready numbers, computed once at startup ---
top_loop = summary_df[summary_df["n_cases"] >= 5].sort_values("mean_loop_rate", ascending=False).head(3)
top_loop_text = ", ".join(f"{r.crop} ({r.mean_loop_rate:.0%})" for r in top_loop.itertuples())
n_cases_total = loop_df["case_id"].nunique()
overall_mismatch_pct = (mismatch_df["is_mismatch"].mean() * 100) if mismatch_df is not None else None

app = Dash(__name__)
app.title = "KCC Process Mining"
server = app.server  # exposed for gunicorn / hosting platforms

COLORS = {
    "bg": "#f5f6fa", "card": "#ffffff", "accent": "#4f46e5", "accent2": "#818cf8",
    "text": "#1f2937", "muted": "#6b7280", "border": "#e5e7eb",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card"], "borderRadius": "12px", "padding": "20px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)", "border": f"1px solid {COLORS['border']}",
}
KPI_STYLE = {**CARD_STYLE, "textAlign": "center", "flex": "1", "minWidth": "160px"}
SECTION_TITLE = {"color": COLORS["text"], "marginBottom": "12px", "fontSize": "18px", "fontWeight": "700"}


def kpi_tile(value, label, color=COLORS["accent"]):
    return html.Div([
        html.Div(value, style={"fontSize": "28px", "fontWeight": "800", "color": color}),
        html.Div(label, style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "4px"}),
    ], style=KPI_STYLE)


def truncate(text, n=140):
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= n else text[:n].rstrip() + "…"


rq1_finding = None
if rq1:
    rq1_finding = html.Li([
        html.B(f"{rq1['winner']} wins on the hand-verified gold set"),
        f" — {rq1['baseline_issuetype_accuracy']:.1%} query-type accuracy (baseline classifier) vs "
        f"{rq1['llm_issuetype_accuracy']:.1%} (LLM extraction), n={rq1['n_compared']} hand-labeled calls. "
        f"The LLM's low macro-F1 ({rq1['llm_issuetype_f1_macro']:.2f}) suggests it collapses many of "
        f"the 42 query types into a handful of common ones, while it does better at judging call "
        f"resolution specifically ({rq1['llm_resolution_signal_accuracy']:.1%} accuracy)."
    ])

app.layout = html.Div(style={
    "fontFamily": "-apple-system, Segoe UI, Arial, sans-serif", "backgroundColor": COLORS["bg"],
    "minHeight": "100vh", "padding": "28px 32px",
}, children=[

    html.Div([
        html.Div([
            html.H1("KCC Advisory Process Mining", style={"margin": 0, "color": COLORS["text"], "fontSize": "28px"}),
            html.Div("KERALA · 2024", style={
                "backgroundColor": COLORS["accent"], "color": "white", "fontSize": "12px", "fontWeight": "700",
                "padding": "4px 10px", "borderRadius": "999px", "letterSpacing": "0.5px", "marginLeft": "12px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div("Semantic-aware LLM process mining over India's Kisan Call Centre helpline. "
                 "This is a single-state deep dive, not a national survey — scoped to Kerala "
                 "on purpose, using real 2024 data, so every number here is grounded in an "
                 "actual verified call rather than a thin slice of five states at once. "
                 "Shows where advisory calls loop, and where AI-generated answers actually "
                 "match what farmers asked.",
                 style={"color": COLORS["muted"], "marginTop": "6px", "fontSize": "14px", "maxWidth": "820px"}),
    ], style={"marginBottom": "22px"}),

    # ---- glossary: what the terms on this page actually mean ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px", "backgroundColor": COLORS["bg"],
                     "border": f"1px dashed {COLORS['border']}"}, children=[
        html.Div("How to read this dashboard", style={**SECTION_TITLE, "fontSize": "15px"}),
        html.Ul([
            html.Li([html.B("Event / call: "), "one farmer phone call, automatically sorted into a "
                     "category (e.g. Plant Protection, Nutrient Management)."]),
            html.Li([html.B("Case: "), "every call about one crop in one district over the year — "
                     "a stand-in for \"one farmer's journey\" since the raw data has no farmer ID "
                     "(a disclosed limitation, not an oversight)."]),
            html.Li([html.B("Loop rate: "), "the share of a case's calls that repeat a category "
                     "already seen earlier in that same case — a proxy for \"this problem kept "
                     "coming back.\""]),
            html.Li([html.B("Duration: "), "days between a case's first and last call."]),
            html.Li([html.B("Variant: "), "the exact sequence of categories a case followed. Two "
                     "cases with the identical sequence share a variant — this is how you spot a "
                     "common advisory pathway rather than a one-off."]),
        ], style={"lineHeight": "1.8", "color": COLORS["text"], "marginBottom": 0, "fontSize": "13px"}),
    ]),

    # ---- headline findings, in plain English ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px", "borderLeft": f"4px solid {COLORS['accent']}"}, children=[
        html.Div("Key findings", style=SECTION_TITLE),
        html.Ul([
            html.Li([html.B(f"{n_cases_total} advisory cases"),
                     f" traced across Kerala districts and crops, with a mean repeat-call "
                     f"('loop') rate of {loop_df['loop_rate'].mean():.0%} — farmers frequently "
                     f"call back about the same category of problem."]),
            html.Li([html.B(f"{top_loop_text}"), " show the highest loop rates among crops with "
                     "meaningful call volume, flagging where advisory quality likely needs the "
                     "most attention."]),
            rq1_finding,
            html.Li([html.B(f"{overall_mismatch_pct:.1f}% of answers flagged"),
                     " as a lexical query-answer mismatch — evidence that surface word-overlap can't "
                     "reliably judge answer quality, motivating the semantic (NLI) approach."])
            if overall_mismatch_pct is not None else None,
        ], style={"lineHeight": "1.9", "color": COLORS["text"], "marginBottom": 0}),
    ]),

    # ---- filters ----
    # (no state filter: every case here is Kerala, so a state dropdown with a
    # single possible value would just be dead UI, not a real filter)
    html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "20px"}, children=[
        dcc.Dropdown(id="crop-filter", options=[{"label": c, "value": c} for c in CROPS],
                     multi=True, placeholder="Filter by crop", style={"flex": "1"}),
    ]),

    # ---- KPI row ----
    html.Div(id="kpi-row", style={"display": "flex", "gap": "16px", "marginBottom": "22px", "flexWrap": "wrap"}),

    # ---- loop rate chart (full width, top N only, sorted) ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Loop rate by crop", style=SECTION_TITLE),
        html.Div(id="loop-rate-note", style={"color": COLORS["muted"], "fontSize": "13px",
                                              "marginBottom": "10px"}),
        dcc.Graph(id="loop-rate-chart", config={"displayModeBar": False}),
    ]),

    # ---- RQ3: does higher answer-mismatch track with higher repeat-issue looping? ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Answer quality vs. repeat calls, by crop", style=SECTION_TITLE),
        html.Div("Each point is one crop. Loop rate is how often the same issue category "
                  "recurs for a crop; mismatch rate is how often the answer given doesn't "
                  "actually address the question asked. If the two track together, answer quality "
                  "may be part of why farmers call back. If they diverge, the two methods are "
                  "catching different failure modes — worth saying explicitly either way.",
                  style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "14px"}),
        html.Div(id="rq3-note", style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "10px"}),
        dcc.Graph(id="rq3-scatter", config={"displayModeBar": False}),
    ]),

    # ---- process explorer: the discovered process map, two ways to view it ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Process explorer", style=SECTION_TITLE),
        html.Div("Automatically discovered from every call in the dataset (not hand-drawn). Boxes are "
                  "issue types, arrows are call-to-call transitions weighted by frequency, and a "
                  "self-loop (an arrow from a box back to itself) means the same issue type recurring "
                  "back-to-back. Filtered to the busiest issue types/paths so the picture stays "
                  "readable — the full unfiltered graph is genuinely unreadable at this scale, which "
                  "is disclosed in the report rather than hidden. Hover over any box or arrow for the "
                  "exact numbers behind it.",
                  style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "14px"}),
        dcc.Dropdown(
            id="map-toggle",
            options=[
                {"label": "Simple map (recommended — interactive, call-frequency flow)", "value": "dfg"},
                {"label": "Formal process model (Inductive Miner Petri net)", "value": "petri"},
            ],
            value="dfg", clearable=False,
            style={"marginBottom": "14px", "fontSize": "14px", "maxWidth": "480px"},
        ),
        html.Div(id="process-map-note", style={"color": COLORS["muted"], "fontSize": "13px",
                                                 "marginBottom": "10px"}),
        html.Div(id="process-map-legend"),
        html.Div(id="process-map-dfg-wrapper", children=[
            dcc.Graph(id="process-map-interactive", config={"displayModeBar": False}),
        ]),
        html.Div(id="process-map-petri-wrapper", children=[
            dcc.Graph(id="process-map-petri-graph", config={"displayModeBar": False}),
        ]),
    ]),

    # ---- variant explorer: which advisory PATTERNS recur across many cases ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Variant explorer", style=SECTION_TITLE),
        html.Div("The case explorer below shows one case's real history. This is the opposite view: "
                  "which exact sequences of categories show up again and again ACROSS cases. A short "
                  "bar means a rare, one-off pattern; a long bar is a well-worn advisory pathway worth "
                  "standardizing guidance for.",
                  style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "14px"}),
        dcc.Graph(id="variant-chart", config={"displayModeBar": False}),
        html.Div("Top variants in detail:", style={"fontWeight": "700", "margin": "16px 0 8px",
                                                     "color": COLORS["text"]}),
        dash_table.DataTable(
            id="variant-table", page_size=8,
            style_cell={"textAlign": "left", "whiteSpace": "normal", "maxWidth": "420px",
                        "fontFamily": "-apple-system, Segoe UI, Arial, sans-serif", "fontSize": "13px",
                        "padding": "8px"},
            style_header={"backgroundColor": COLORS["bg"], "fontWeight": "700"},
            style_table={"overflowX": "auto"},
        ),
    ]),

    # ---- case summary table (no raw variant text -- just the scannable numbers) ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Busiest cases (most events = most back-and-forth)", style=SECTION_TITLE),
        html.Div("Pick any case in the explorer below to see its actual call-by-call history.",
                  style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "10px"}),
        dash_table.DataTable(
            id="case-summary-table", page_size=8,
            style_cell={"textAlign": "left", "fontFamily": "-apple-system, Segoe UI, Arial, sans-serif",
                        "fontSize": "13px", "padding": "8px"},
            style_header={"backgroundColor": COLORS["bg"], "fontWeight": "700"},
            style_table={"overflowX": "auto"},
        ),
    ]),

    # ---- interactive case explorer ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("Case explorer — walk through a single case's real call history", style=SECTION_TITLE),
        html.Div("A \"case\" here is every call about one crop in one district over the year "
                 "(not a single farmer's journey — a known limitation, see report). Pick one below "
                 "to see its calls in order, colored by category, with the actual query/answer text.",
                 style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "14px"}),
        dcc.Dropdown(id="case-select", options=CASE_OPTIONS, value=DEFAULT_CASE, clearable=False,
                     placeholder="Select a case", style={"marginBottom": "14px"}),
        html.Div(id="case-stats-row", style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                                              "flexWrap": "wrap"}),
        dcc.Graph(id="case-timeline", config={"displayModeBar": False}),
        html.Div("Call-by-call detail (in order):", style={"fontWeight": "700", "margin": "16px 0 8px",
                                                             "color": COLORS["text"]}),
        dash_table.DataTable(
            id="case-detail-table", page_size=10,
            style_cell={"textAlign": "left", "whiteSpace": "normal", "maxWidth": "360px",
                        "fontFamily": "-apple-system, Segoe UI, Arial, sans-serif", "fontSize": "13px",
                        "padding": "8px"},
            style_header={"backgroundColor": COLORS["bg"], "fontWeight": "700"},
            style_table={"overflowX": "auto"},
        ),
    ]),

    # ---- LLM extraction sample, if available ----
    html.Div(style={**CARD_STYLE, "marginBottom": "22px"}, children=[
        html.Div("LLM extraction sample (alternate classification method)", style=SECTION_TITLE),
        dash_table.DataTable(
            id="llm-table",
            data=llm_df.head(25).to_dict("records") if llm_df is not None else [],
            columns=[{"name": c, "id": c} for c in llm_df.columns] if llm_df is not None else [],
            page_size=6,
            style_cell={"textAlign": "left", "whiteSpace": "normal", "maxWidth": "320px", "fontSize": "13px"},
            style_header={"backgroundColor": COLORS["bg"], "fontWeight": "700"},
            style_table={"overflowX": "auto"},
        ) if llm_df is not None else html.Div(
            "No LLM extraction output found yet — run src/llm_extraction.py first.",
            style={"color": COLORS["muted"]}),
    ]),

    # ---- live demo ----
    html.Div(style=CARD_STYLE, children=[
        html.Div("Try it — live classifier + retrieval demo", style=SECTION_TITLE),
        html.Div("The classifier only sorts a query into a category — it doesn't invent farming advice, "
                  "and shouldn't (a confidently wrong fertilizer recommendation is worse than none). What "
                  "it can do honestly is show real answers KCC actually gave to similar past questions.",
                  style={"color": COLORS["muted"], "fontSize": "13px", "marginBottom": "14px"}),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            dcc.Textarea(id="live-query", placeholder="Type a farmer query here, e.g. "
                         "'coconut tree leaves turning yellow, what fertilizer should I use'",
                         style={"flex": "1", "minWidth": "280px", "height": "90px", "borderRadius": "8px",
                                "border": f"1px solid {COLORS['border']}", "padding": "10px"}),
            html.Div(id="live-query-result", style={
                "flex": "1", "minWidth": "220px", "padding": "12px", "backgroundColor": COLORS["bg"],
                "borderRadius": "8px", "color": COLORS["text"]}),
        ]),
        html.Div("Similar real past cases (real KCC answers, retrieved by text similarity):",
                  style={"fontWeight": "700", "margin": "16px 0 8px", "color": COLORS["text"]}),
        dash_table.DataTable(
            id="live-query-similar", page_size=5,
            style_cell={"textAlign": "left", "whiteSpace": "normal", "maxWidth": "360px",
                        "fontFamily": "-apple-system, Segoe UI, Arial, sans-serif", "fontSize": "13px",
                        "padding": "8px"},
            style_header={"backgroundColor": COLORS["bg"], "fontWeight": "700"},
            style_table={"overflowX": "auto"},
        ),
    ]),
])


@app.callback(
    Output("kpi-row", "children"),
    Output("loop-rate-chart", "figure"),
    Output("loop-rate-note", "children"),
    Output("case-summary-table", "data"),
    Output("case-summary-table", "columns"),
    Output("rq3-scatter", "figure"),
    Output("rq3-note", "children"),
    Input("crop-filter", "value"),
)
def update_dashboard(crops):
    ldf, ddf, vdf, mdf = loop_df, dur_df, var_df, mismatch_df
    if crops:
        ldf = ldf[ldf["crop"].isin(crops)]
        ddf = ddf[ddf["crop"].isin(crops)]
        vdf = vdf[vdf["crop"].isin(crops)]
        if mdf is not None:
            mdf = mdf[mdf["Crop"].isin(crops)]

    n_cases = ldf["case_id"].nunique()
    mean_loop = ldf["loop_rate"].mean() if len(ldf) else 0
    mean_dur = ddf["duration_days"].mean() if len(ddf) else 0
    pct_mismatch = (mdf["is_mismatch"].mean() * 100) if mdf is not None and len(mdf) else None

    kpis = [
        kpi_tile(f"{n_cases}", "Cases"),
        kpi_tile(f"{mean_loop:.2f}", "Mean loop rate"),
        kpi_tile(f"{mean_dur:.0f} days", "Mean case duration"),
    ]
    if pct_mismatch is not None:
        kpis.append(kpi_tile(f"{pct_mismatch:.1f}%", "Flagged mismatches", color="#dc2626"))

    # top 15 crops by loop rate, restricted to crops with enough cases to be meaningful --
    # but if the current selection doesn't even have 5 cases in ANY crop (e.g. one small
    # crop picked in the filter), fall back to showing what's actually there instead of an
    # empty chart, and say so explicitly rather than leaving the old "5+ cases" caption up
    max_cases = ldf.groupby("crop")["case_id"].nunique().max() if len(ldf) else 0
    threshold = min(5, max_cases) if max_cases else 1
    crop_stats = ldf.groupby("crop").agg(loop_rate=("loop_rate", "mean"),
                                          n_cases=("case_id", "nunique")).reset_index()
    crop_stats = crop_stats[crop_stats["n_cases"] >= threshold]
    crop_stats = crop_stats.sort_values("loop_rate", ascending=True).tail(15)
    if threshold < 5:
        note = (f"This selection has no crop with 5+ cases, so showing all {len(crop_stats)} crop(s) "
                 f"in the current filter instead.")
    else:
        note = f"Crops with 5+ cases, top 15 of {len(crop_stats)} shown."

    if len(crop_stats) == 0:
        fig = go.Figure()
        fig.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="white",
                           annotations=[dict(text="No cases match this filter.", showarrow=False,
                                              font=dict(color=COLORS["muted"]))])
    else:
        fig = px.bar(crop_stats, x="loop_rate", y="crop", orientation="h",
                     color_discrete_sequence=[COLORS["accent"]],
                     text=crop_stats["loop_rate"].map(lambda v: f"{v:.2f}"),
                     labels={"loop_rate": "Mean loop rate", "crop": ""})
        # a genuinely-zero loop rate (e.g. every case in this selection had only
        # one call, so no repeat was even possible) draws a zero-length bar --
        # without a fixed axis range, Plotly auto-scales a single all-zero bar
        # to a nonsensical [-1, 1] range and the bar vanishes entirely, looking
        # like a rendering bug rather than an honest "the rate really is 0"
        max_rate = max(crop_stats["loop_rate"].max(), 0.05)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(range=[0, max_rate * 1.25])
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor="white", paper_bgcolor="white", height=420)

    case_summary = ldf.merge(ddf[["case_id", "duration_days"]], on="case_id", how="left")
    case_summary = case_summary.sort_values("n_events", ascending=False)
    cols = ["case_id", "n_events", "loop_rate", "duration_days", "crop", "state"]
    case_summary = case_summary[cols].copy()
    case_summary["loop_rate"] = case_summary["loop_rate"].round(2)
    table_data = case_summary.to_dict("records")
    table_cols = [{"name": c, "id": c} for c in cols]

    # ---- RQ3 scatter: per-crop loop rate (RQ2) vs. per-crop mismatch rate (RQ3) ----
    if mdf is None:
        rq3_fig = go.Figure()
        rq3_fig.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="white",
                               annotations=[dict(text="No semantic mismatch output found — run "
                                                       "src/semantic_mismatch.py first.",
                                                  showarrow=False, font=dict(color=COLORS["muted"]))])
        rq3_note = ""
    else:
        loop_by_crop = ldf.groupby("crop").agg(loop_rate=("loop_rate", "mean"),
                                                 n_cases=("case_id", "nunique")).reset_index()
        mismatch_by_crop = mdf.groupby("Crop").agg(mismatch_rate=("is_mismatch", "mean"),
                                                     n_calls=("is_mismatch", "count")).reset_index()
        rq3_df = loop_by_crop.merge(mismatch_by_crop, left_on="crop", right_on="Crop", how="inner")
        # tiny crops (a handful of calls) make a noisy mismatch_rate that isn't a real signal --
        # same reasoning as the loop-rate chart's threshold above, applied here too
        rq3_df = rq3_df[rq3_df["n_calls"] >= 5]

        if len(rq3_df) == 0:
            rq3_fig = go.Figure()
            rq3_fig.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="white",
                                   annotations=[dict(text="No crop in this selection has enough calls "
                                                           "(5+) for a meaningful mismatch rate.",
                                                      showarrow=False, font=dict(color=COLORS["muted"]))])
            rq3_note = ""
        else:
            rq3_fig = px.scatter(rq3_df, x="loop_rate", y="mismatch_rate", size="n_cases",
                                  hover_name="crop", color_discrete_sequence=[COLORS["accent"]],
                                  labels={"loop_rate": "Mean loop rate",
                                          "mismatch_rate": "Answer-mismatch rate"},
                                  hover_data={"n_cases": True, "n_calls": True})
            rq3_fig.update_traces(marker=dict(line=dict(width=1, color="white")))
            rq3_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor="white", paper_bgcolor="white", height=420)
            corr = rq3_df["loop_rate"].corr(rq3_df["mismatch_rate"])
            rq3_note = (f"Crops with 5+ calls, n={len(rq3_df)}. Correlation between loop rate and "
                        f"mismatch rate across these crops: {corr:.2f} "
                        f"({'the two track together' if corr > 0.3 else 'weak/no relationship' if abs(corr) < 0.3 else 'they move in opposite directions'} "
                        f"in this selection).")

    return kpis, fig, note, table_data, table_cols, rq3_fig, rq3_note


def _wrap_label(name, max_len=16):
    """Break a long issue-type name onto two lines at the nearest space to
    the middle, so it doesn't get clipped or run into neighboring nodes."""
    if len(name) <= max_len or " " not in name:
        return name
    mid = len(name) // 2
    space_positions = [i for i, ch in enumerate(name) if ch == " "]
    best = min(space_positions, key=lambda i: abs(i - mid))
    return name[:best] + "<br>" + name[best + 1:]


def build_dfg_figure(dfg, start_activities, end_activities, node_totals):
    """Builds an interactive Plotly version of the directly-follows graph --
    same discovery/trimming as the static report figure, but hoverable exact
    counts instead of a flat graphviz image.

    Design choices made after actually looking at a first draft of this and
    finding it unreadable: full names go BELOW each node (never crammed
    inside the circle, which clipped text badly at small sizes); only the
    call count sits inside the marker; and when both A->B and B->A exist
    (very common here -- issue types ping-pong back and forth) the two
    arrows are offset to either side of the straight line between them,
    otherwise they visually merge into one thick double-ended blob.
    """
    nodes = sorted(node_totals, key=lambda n: -node_totals[n])
    self_loops = {n: dfg.get((n, n), 0) for n in nodes}
    edges = [(a, b, c) for (a, b), c in dfg.items() if a != b]
    edge_set = {(a, b) for a, b, _ in edges}

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for a, b, c in edges:
        G.add_edge(a, b, weight=c)
    # circular layout: guarantees even spacing regardless of how connected a
    # node is, which matters more here than graph-theoretically "meaningful"
    # clustering for a non-technical viewer
    pos = nx.circular_layout(G, scale=1.3)

    max_total = max(node_totals.values()) if node_totals else 1
    max_edge = max((c for _, _, c in edges), default=1)

    fig = go.Figure()

    annotations = []
    for a, b, c in sorted(edges, key=lambda e: e[2]):
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        # if the reverse edge also exists, nudge this line sideways
        # (perpendicular to its direction) so the two arrows run in
        # parallel instead of stacking on top of each other
        if (b, a) in edge_set:
            dx, dy = x1 - x0, y1 - y0
            length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
            perp_x, perp_y = -dy / length, dx / length
            offset = 0.06
            x0, y0 = x0 + perp_x * offset, y0 + perp_y * offset
            x1, y1 = x1 + perp_x * offset, y1 + perp_y * offset

        width = 1.5 + 6.5 * (c / max_edge)
        annotations.append(dict(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=width,
            arrowcolor=COLORS["accent2"], standoff=24, startstandoff=24, opacity=0.75,
        ))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_trace(go.Scatter(
            x=[mx], y=[my], mode="markers+text", text=[f"{c}"],
            textfont=dict(size=12, color=COLORS["accent"], family="Arial Black, Arial, sans-serif"),
            marker=dict(size=16, color="white", opacity=0.01),
            hovertemplate=f"<b>{a} → {b}</b><br>{c} calls followed this path<extra></extra>",
            showlegend=False,
        ))

    node_colors = [COLORS["muted"] if n == "Other" else COLORS["accent"] for n in nodes]
    node_sizes = [34 + 46 * ((node_totals[n] / max_total) ** 0.5) for n in nodes]
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    hover_texts, name_labels, count_labels = [], [], []
    for n in nodes:
        loop_txt = (f"<br>{self_loops[n]} repeat calls (same issue twice in a row)"
                    if self_loops[n] else "")
        start_txt = f"<br>Starts {start_activities[n]} case(s)" if n in start_activities else ""
        end_txt = f"<br>Ends {end_activities[n]} case(s)" if n in end_activities else ""
        hover_texts.append(f"<b>{n}</b><br>{node_totals[n]} calls total{loop_txt}{start_txt}{end_txt}")
        loop_badge = f"<br>({self_loops[n]} repeats)" if self_loops[n] else ""
        name_labels.append(_wrap_label(n) + loop_badge)
        count_labels.append(f"{node_totals[n]}")

    # name + repeat-count badge, always visible BELOW the node -- pushed down
    # by a bit more than each node's own radius (converted from marker px to
    # roughly the same data-coordinate scale as the axis range below) so the
    # label never overlaps the circle it belongs to, even for the biggest nodes
    label_y = [y - (size / 2 + 10) / 145 for y, size in zip(node_y, node_sizes)]
    fig.add_trace(go.Scatter(
        x=node_x, y=label_y, mode="text", text=name_labels,
        textposition="bottom center", textfont=dict(size=12, color=COLORS["text"]),
        hoverinfo="skip", showlegend=False,
    ))
    # the marker itself, with just the raw call count inside (always short,
    # always fits) and the full detail on hover
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=count_labels,
        textposition="middle center", textfont=dict(size=13, color="white", family="Arial Black, Arial"),
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color="white")),
        hovertext=hover_texts, hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        annotations=annotations, showlegend=False, height=560,
        margin=dict(l=30, r=30, t=20, b=60), plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-1.9, 1.9]), yaxis=dict(visible=False, range=[-1.8, 1.9]),
        hovermode="closest",
    )
    return fig


def build_petri_figure(nodes, edges):
    """Interactive version of the formal Petri net, positioned using
    graphviz's own `dot` layout (see extract_petri_layout) rather than a
    from-scratch layout -- `dot` already does a good job of laying out a
    left-to-right control-flow diagram, so there's no reason to reinvent it.
    Circles are places, squares are transitions (black = silent/routing
    only, colored = an actual call category), matching the shape key.
    """
    xs = [n["x"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()]
    x_pad = max((max(xs) - min(xs)) * 0.06, 0.8) if xs else 1
    y_pad = max((max(ys) - min(ys)) * 0.15, 0.8) if ys else 1

    x_range = [min(xs) - x_pad, max(xs) + x_pad] if xs else [-1, 1]
    y_range = [min(ys) - y_pad, max(ys) + y_pad] if ys else [-1, 1]
    aspect = (x_range[1] - x_range[0]) / max(y_range[1] - y_range[0], 0.1)
    height = int(max(320, min(620, 900 / max(aspect, 1))))
    # px-per-data-unit on the y axis, so the label offset below a transition
    # box is calibrated to actually clear the marker (half its pixel size,
    # plus a small gap) regardless of how tall/dense this particular subset's
    # graph is -- a fixed multiple of y_pad overcorrected on dense graphs
    px_per_unit_y = height / (y_range[1] - y_range[0])
    label_offset = (26 / 2 + 9) / px_per_unit_y

    fig = go.Figure()
    annotations = []
    for a, b in edges:
        if a not in nodes or b not in nodes:
            continue
        na, nb = nodes[a], nodes[b]
        annotations.append(dict(
            x=nb["x"], y=nb["y"], ax=na["x"], ay=na["y"], xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
            arrowcolor=COLORS["muted"], standoff=16, startstandoff=16, opacity=0.7,
        ))

    silent = [n for n in nodes.values() if n["kind"] == "transition" and n["is_silent"]]
    labeled = [n for n in nodes.values() if n["kind"] == "transition" and not n["is_silent"]]
    start_p = [n for n in nodes.values() if n["kind"] == "place" and n["is_start"]]
    end_p = [n for n in nodes.values() if n["kind"] == "place" and n["is_end"]]
    plain_p = [n for n in nodes.values() if n["kind"] == "place" and not n["is_start"] and not n["is_end"]]

    def add_group(group, size, color, symbol, text=None, textcolor="white", hover_fn=None):
        if not group:
            return
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in group], y=[n["y"] for n in group],
            mode="markers+text" if text else "markers",
            text=text, textposition="middle center", textfont=dict(size=11, color=textcolor),
            marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1.5, color=COLORS["muted"])),
            hovertext=[hover_fn(n) for n in group] if hover_fn else None,
            hoverinfo="text" if hover_fn else "skip", showlegend=False,
        ))

    add_group(plain_p, 14, "white", "circle",
              hover_fn=lambda n: "Waiting point between steps — not a real event, "
                                  "just formal bookkeeping the algorithm needs.")
    add_group(start_p, 24, COLORS["accent"], "circle", text=["S"] * len(start_p),
              hover_fn=lambda n: "Start of the process (initial marking).")
    add_group(end_p, 26, COLORS["text"], "circle", text=["E"] * len(end_p),
              hover_fn=lambda n: "End of the process (final marking).")
    add_group(silent, 13, "black", "square",
              hover_fn=lambda n: "Silent step — routing logic the miner needs, not an actual call.")

    if labeled:
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in labeled], y=[n["y"] for n in labeled], mode="markers",
            marker=dict(size=26, color=COLORS["accent"], symbol="square",
                        line=dict(width=1.5, color=COLORS["muted"])),
            hovertext=[f"<b>{n['label']}</b><br>An actual call in this flow." for n in labeled],
            hoverinfo="text", showlegend=False,
        ))
        label_y = [n["y"] - label_offset for n in labeled]
        fig.add_trace(go.Scatter(
            x=[n["x"] for n in labeled], y=label_y, mode="text",
            text=[_wrap_label(n["label"], max_len=14) for n in labeled],
            textposition="bottom center", textfont=dict(size=11, color=COLORS["text"]),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        annotations=annotations, showlegend=False, height=height,
        margin=dict(l=20, r=20, t=20, b=50), plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range, scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )
    return fig


PETRI_LEGEND = html.Div([
    html.Div("Shape key (this is formal notation, not the recommended view — hover any shape for detail):",
             style={"fontWeight": "700", "fontSize": "13px", "marginBottom": "6px", "color": COLORS["text"]}),
    html.Ul([
        html.Li([html.B("Circle marked \"S\": "), "the process starts here (the initial marking)."]),
        html.Li([html.B("Colored square: "), "an actual call category happening (a \"transition\") — "
                 "the name is labeled underneath it."]),
        html.Li([html.B("Solid black square: "), "a silent step — routing logic the miner needs to "
                 "represent choice/parallelism, not an actual call."]),
        html.Li([html.B("Plain empty circle: "), "a waiting point between steps — not a real event itself, "
                 "just formal bookkeeping the algorithm needs."]),
        html.Li([html.B("Circle marked \"E\": "), "the process ends here (the final marking)."]),
    ], style={"fontSize": "13px", "color": COLORS["muted"], "lineHeight": "1.7", "marginBottom": "12px"}),
], style={"backgroundColor": COLORS["bg"], "border": f"1px dashed {COLORS['border']}",
          "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "12px"})


DFG_VISIBLE = {"display": "block"}
DFG_HIDDEN = {"display": "none"}


@app.callback(
    Output("process-map-interactive", "figure"),
    Output("process-map-petri-graph", "figure"),
    Output("process-map-dfg-wrapper", "style"),
    Output("process-map-petri-wrapper", "style"),
    Output("process-map-note", "children"),
    Output("process-map-legend", "children"),
    Input("map-toggle", "value"),
    Input("crop-filter", "value"),
)
def update_process_map(which, crops):
    legend = PETRI_LEGEND if which == "petri" else None
    dfg_style = DFG_VISIBLE if which == "dfg" else DFG_HIDDEN
    petri_style = DFG_HIDDEN if which == "dfg" else DFG_VISIBLE
    empty_fig = go.Figure()
    empty_fig.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="white")

    # Both views are generated live from the current data every time this
    # callback fires -- no pre-rendered picture, cached or otherwise, and
    # neither is a flat image anymore: both render as hoverable Plotly
    # figures built straight from what pm4py just discovered. Timed this at
    # ~0.3s for the Petri net and well under that for the DFG even on the
    # full unfiltered dataset (6570 calls), so no cached shortcut is needed.
    if not crops:
        subset = raw_df
    else:
        subset = raw_df[raw_df["Crop"].isin(crops)]

    n_calls = len(subset)
    n_cases = subset["case_id"].nunique()

    if n_calls < 2:
        note = (f"Only {n_calls} call{'s' if n_calls != 1 else ''} for this selection — there's no "
                 f"call-to-call transition to draw a flow from. See the KPI tiles and case explorer "
                 f"above for what that one call actually was.")
        return empty_fig, empty_fig, dfg_style, petri_style, note, None

    log_df = build_event_log_df(subset, activity_col=ACTIVITY_COL, date_col="CreatedOn")
    note = (f"Auto-generated live from {n_calls} calls across {n_cases} case(s) — every crop, full "
             f"dataset." if not crops else
            f"Auto-generated live from {n_calls} calls across {n_cases} case(s) matching this filter.")

    if which == "dfg":
        try:
            dfg, starts, ends, totals = discover_dfg_data(log_df)
            fig = build_dfg_figure(dfg, starts, ends, totals)
            return fig, empty_fig, dfg_style, petri_style, note, legend
        except Exception:
            note = (f"Couldn't build a process diagram from this selection ({n_calls} calls, "
                     f"{n_cases} case(s)) — too little structure for the miner to draw a flow from. "
                     f"Try a broader crop selection.")
            return empty_fig, empty_fig, dfg_style, petri_style, note, None

    # Petri net view -- always mined fresh from log_df, rendered as an
    # interactive figure (see build_petri_figure), never a static file
    try:
        _, _, net, im, fm = discover_model(log_df)
        nodes, edges = extract_petri_layout(net, im, fm)
        fig = build_petri_figure(nodes, edges)
        return empty_fig, fig, dfg_style, petri_style, note, legend
    except Exception:
        note = (f"Couldn't build a process diagram from this selection ({n_calls} calls, "
                 f"{n_cases} case(s)) — too little structure for the miner to draw a flow from. "
                 f"Try a broader crop selection.")
        return empty_fig, empty_fig, dfg_style, petri_style, note, None


@app.callback(
    Output("variant-chart", "figure"),
    Output("variant-table", "data"),
    Output("variant-table", "columns"),
    Input("crop-filter", "value"),
)
def update_variant_explorer(crops):
    vc = variant_counts
    vdf_scope = var_df
    if crops:
        vdf_scope = vdf_scope[vdf_scope["crop"].isin(crops)]
        vc = (vdf_scope.groupby("variant")
              .agg(n_cases=("case_id", "nunique"),
                   example_crops=("crop", lambda s: ", ".join(sorted(set(s))[:3])))
              .reset_index())
        vc["n_steps"] = vc["variant"].str.count("->") + 1
        total = vdf_scope["case_id"].nunique()
        vc["pct_of_cases"] = (vc["n_cases"] / total * 100).round(1) if total else 0
        vc = vc.sort_values("n_cases", ascending=False).reset_index(drop=True)
        vc.insert(0, "rank", range(1, len(vc) + 1))

    top15 = vc.head(15).copy()
    top15["short_label"] = [f"#{r.rank} ({r.n_steps} step{'s' if r.n_steps != 1 else ''})"
                             for r in top15.itertuples()]
    fig = px.bar(top15.sort_values("n_cases"), x="n_cases", y="short_label", orientation="h",
                 color_discrete_sequence=[COLORS["accent2"]],
                 hover_data={"variant": True, "example_crops": True},
                 labels={"n_cases": "Number of cases following this exact sequence", "short_label": ""})
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="white", paper_bgcolor="white", height=420)

    table = vc.head(30).copy()
    table["variant"] = table["variant"].apply(lambda t: truncate(t, 200))
    table = table.rename(columns={"rank": "Rank", "n_cases": "# cases", "pct_of_cases": "% of cases",
                                   "n_steps": "Steps", "example_crops": "Example crops",
                                   "variant": "Sequence"})
    table = table[["Rank", "# cases", "% of cases", "Steps", "Example crops", "Sequence"]]
    table_cols = [{"name": c, "id": c} for c in table.columns]

    return fig, table.to_dict("records"), table_cols


@app.callback(
    Output("case-select", "options"),
    Output("case-select", "value"),
    Input("crop-filter", "value"),
)
def update_case_options(crops):
    df = case_options_df
    if crops:
        df = df[df["crop"].isin(crops)]
    options = [
        {"label": f"{r.case_id}  —  {r.n_events} events, loop rate {r.loop_rate:.0%}", "value": r.case_id}
        for r in df.itertuples()
    ]
    default = df.iloc[0]["case_id"] if len(df) else None
    return options, default


@app.callback(
    Output("case-stats-row", "children"),
    Output("case-timeline", "figure"),
    Output("case-detail-table", "data"),
    Output("case-detail-table", "columns"),
    Input("case-select", "value"),
)
def update_case_explorer(case_id):
    empty_fig = go.Figure()
    empty_fig.update_layout(height=200, plot_bgcolor="white", paper_bgcolor="white")
    if not case_id:
        return [], empty_fig, [], []

    events = raw_df[raw_df["case_id"] == case_id].sort_values("CreatedOn").reset_index(drop=True)
    if len(events) == 0:
        return [html.Div("No events found for this case.", style={"color": COLORS["muted"]})], empty_fig, [], []

    loop_row = loop_df[loop_df["case_id"] == case_id]
    dur_row = dur_df[dur_df["case_id"] == case_id]
    loop_rate = float(loop_row["loop_rate"].iloc[0]) if len(loop_row) else None
    duration = int(dur_row["duration_days"].iloc[0]) if len(dur_row) else None

    stats = [
        kpi_tile(f"{len(events)}", "Calls in this case"),
        kpi_tile(f"{loop_rate:.0%}" if loop_rate is not None else "—", "Loop rate"),
        kpi_tile(f"{duration} days" if duration is not None else "—", "Duration"),
        kpi_tile(f"{events['Crop'].iloc[0]}", "Crop", color=COLORS["muted"]),
        kpi_tile(f"{events['DistrictName'].iloc[0]}", "District", color=COLORS["muted"]),
    ]

    events["step"] = range(1, len(events) + 1)
    events["hover_query"] = events["QueryText"].apply(lambda t: truncate(t, 100))
    fig = px.scatter(
        events, x="CreatedOn", y=[1] * len(events), color=ACTIVITY_COL,
        hover_data={"step": True, "hover_query": True, "CreatedOn": True, ACTIVITY_COL: True},
        labels={"CreatedOn": "Call date", ACTIVITY_COL: "Issue type"},
    )
    fig.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    # thin connecting line to show the sequence, drawn behind the colored points
    fig.add_trace(go.Scatter(x=events["CreatedOn"], y=[1] * len(events), mode="lines",
                              line=dict(color=COLORS["border"], width=1), showlegend=False, hoverinfo="skip"))
    fig.data = (fig.data[-1],) + fig.data[:-1]  # push the line to the back
    fig.update_yaxes(visible=False, range=[0.5, 1.5])
    fig.update_layout(height=220, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=10, r=10, t=10, b=10), legend_title_text="Issue type")

    detail = events[["CreatedOn", ACTIVITY_COL, "QueryText", "KccAns"]].copy()
    detail["CreatedOn"] = detail["CreatedOn"].dt.strftime("%Y-%m-%d")
    detail["QueryText"] = detail["QueryText"].apply(lambda t: truncate(t, 160))
    detail["KccAns"] = detail["KccAns"].apply(lambda t: truncate(t, 160))
    detail = detail.rename(columns={"CreatedOn": "Date", ACTIVITY_COL: "Issue type",
                                     "QueryText": "Query", "KccAns": "Answer"})
    detail_cols = [{"name": c, "id": c} for c in detail.columns]

    return stats, fig, detail.to_dict("records"), detail_cols


def _prediction_card(label, value, confidence=None):
    conf_bar = None
    if confidence is not None:
        pct = f"{confidence * 100:.0f}%"
        conf_bar = html.Div([
            html.Div(style={"width": f"{confidence * 100:.0f}%", "height": "6px",
                             "backgroundColor": COLORS["accent"], "borderRadius": "3px"}),
        ], style={"width": "100%", "height": "6px", "backgroundColor": COLORS["border"],
                   "borderRadius": "3px", "marginTop": "8px"})
        conf_label = html.Div(f"Model confidence: {pct}", style={"fontSize": "11px",
                               "color": COLORS["muted"], "marginTop": "4px"})
    fig_children = [
        html.Div(label, style={"fontSize": "12px", "color": COLORS["muted"], "fontWeight": "600",
                                "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        html.Div(value, style={"fontSize": "20px", "fontWeight": "800", "color": COLORS["text"],
                                "marginTop": "4px"}),
    ]
    if conf_bar is not None:
        fig_children += [conf_bar, conf_label]
    return html.Div(fig_children, style={**CARD_STYLE, "padding": "14px 16px", "flex": "1", "minWidth": "180px"})


def _svm_confidence(scores, classes, predicted_label):
    """Turn a LinearSVC decision_function output into a relative-confidence
    readout for the predicted class -- NOT a calibrated probability, so it's
    labeled "model confidence" rather than a percentage-likelihood claim.
    Handles both shapes decision_function can return: one score per class
    (3+ classes, softmax-normalized) and a single scalar (exactly 2 classes,
    sigmoid-normalized, since sklearn only stores one margin for the binary
    case rather than one per class).
    """
    scores = np.atleast_1d(scores)
    if len(classes) == 2 and scores.shape[-1] != len(classes):
        # binary case: one scalar margin, positive score favors classes_[1]
        margin = float(scores[0]) if scores.ndim else float(scores)
        prob_positive = 1 / (1 + np.exp(-margin))
        prob = prob_positive if predicted_label == classes[1] else (1 - prob_positive)
        return float(prob)
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    class_index = {c: i for i, c in enumerate(classes)}
    return float(probs[class_index[predicted_label]])


def _predict_with_confidence(clf, text):
    """Same hierarchical prediction as HierarchicalClassifier.predict(), plus a
    relative-confidence readout at both the category and query-type stage.
    Query-type confidence is only available when the category actually has a
    trained sub-classifier: some categories have too few labelled examples and
    fall back to a stored majority-class string with no model at all, and
    there's no honest confidence number to show for a plain majority-class
    guess, so that case returns confidence=None rather than a fabricated value.
    """
    Xc = clf.category_vectorizer.transform([text])
    cat_scores = clf.category_clf.decision_function(Xc)[0]
    cat_classes = clf.category_clf.classes_
    pred_cat = clf.category_clf.predict(Xc)[0]
    cat_confidence = _svm_confidence(cat_scores, cat_classes, pred_cat)

    vec = clf.querytype_vectorizers.get(pred_cat)
    sub_clf = clf.querytype_clfs.get(pred_cat)
    if vec is None:
        pred_qtype = sub_clf if isinstance(sub_clf, str) else "UNKNOWN"
        qtype_confidence = None
    else:
        Xq = vec.transform([text])
        pred_qtype = sub_clf.predict(Xq)[0]
        qtype_scores = sub_clf.decision_function(Xq)[0]
        qtype_confidence = _svm_confidence(qtype_scores, sub_clf.classes_, pred_qtype)
    return pred_cat, cat_confidence, pred_qtype, qtype_confidence


@app.callback(
    Output("live-query-result", "children"),
    Output("live-query-similar", "data"),
    Output("live-query-similar", "columns"),
    Input("live-query", "value"),
)
def live_query_demo(text):
    if not text or not text.strip():
        return "Type a query above to see the classifier's live prediction.", [], []
    if classifier is None:
        return "Baseline classifier not found — run src/baseline_classifier.py first.", [], []

    pred_cat, cat_confidence, pred_qtype, qtype_confidence = _predict_with_confidence(classifier, text)
    result = html.Div([
        _prediction_card("Predicted category", pred_cat, confidence=cat_confidence),
        _prediction_card("Predicted query type", pred_qtype, confidence=qtype_confidence),
    ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})

    similar = retriever.query(text, k=5)
    similar = similar[similar["similarity"] > 0.03]  # drop near-zero matches -- not actually similar
    if len(similar) == 0:
        cols = [{"name": c, "id": c} for c in ["Note"]]
        return result, [{"Note": "No sufficiently similar past query found in the dataset for this one."}], cols

    similar = similar.copy()
    similar["similarity"] = (similar["similarity"] * 100).round(0).astype(int).astype(str) + "%"
    similar["QueryText"] = similar["QueryText"].apply(lambda t: truncate(t, 140))
    similar["KccAns"] = similar["KccAns"].apply(lambda t: truncate(t, 220))
    similar = similar.rename(columns={"QueryText": "Similar past query", "KccAns": "Real answer given",
                                       "Crop": "Crop", "Category": "Category", "similarity": "Match"})
    similar = similar[["Match", "Crop", "Category", "Similar past query", "Real answer given"]]
    table_cols = [{"name": c, "id": c} for c in similar.columns]

    return result, similar.to_dict("records"), table_cols


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
