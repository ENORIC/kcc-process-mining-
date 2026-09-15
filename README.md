This is the Final Module Project for **B198 – AI Applications for Digital Business**

## Semantic-Aware LLM Process Mining Advisory Workflow
- Detecting Repeat-Issue Loops and Answer-Quality Gaps in India's Kisan Call Centre (Kerala, 2024)
- By — Enosh Paul Niju GH1026595

My initial motivation for this project was curiosity about what actually happens after a
farmer calls a government helpline for advice — not just what they ask, but what happens
next. Does the same farmer call back with the same problem again? Does the advice given
actually match the question that was asked? And can a machine reliably tell what *kind* of
problem it even was, well enough to trust it as the input to the next layer of analysis?

The core question driving this project: **can a text-classification layer, a process-mining
layer, and a semantic-matching layer, stacked together, reveal breakdowns in an advisory
service that no single one of those methods would catch on its own?**

This project is entirely focused on data analysis and applied NLP/process mining, built on
real operational data from Kerala's Kisan Call Centre — 26,601 raw farmer call records from
2024, cleaned down to 6,570 usable records — which made it a genuine dataset to explore
where an advisory pipeline like this earns trust, and where it quietly doesn't.

## Data Source
*Dataset:* Kisan Call Centre (KCC) Farmer Query Data, Kerala
*Source:* https://data.gov.in

*Live dashboard link: -->*
https://kcc-process-mining.onrender.com

- Raw export: `data/raw/kcc_kerala_2024.csv`
- Cleaned records: `data/processed/kcc_clean.csv` (6,570 records after cleaning)
- Hand-labeled gold evaluation set: `data/gold/gold_set_final.csv`
- Trained baseline classifier: `models/baseline_classifier.joblib`

The data was scoped specifically to Kerala for the year 2024 — a deliberate choice for
analytical completeness rather than one made from convenience. Purely administrative
queries (e.g. "Government Schemes," "Training") were dropped during cleaning, since they
are access-to-service questions rather than the agronomic problems the process-mining and
semantic-matching layers are built to evaluate, and records with Malayalam-script answers
were dropped so the classifier and semantic-mismatch layers work against one consistent
language throughout.

## Code Documentation

Every file under `src/` and `dashboard/app.py` is annotated throughout with `#` comments.
Each comment block explains which functions/components are being used in that section and
what it produces.

## Table of Contents
- Overview
- RQ1 — Text Intelligence: Baseline TF-IDF/SVM Classifier vs. LLM Structured Extraction
- RQ2 — Process Mining: Case Construction, Repeat-Issue Loop Rates & Crop-Level Patterns
- RQ3 — Semantic Mismatch: Does the Answer Given Actually Match the Question Asked?
- Interactive Process Explorer — Directly-Follows Graph & Formal (Petri Net) Process Model
- Live Classifier + Retrieval Demo
- Honest Deviations From the Original Proposal
- Project Structure & How to Run It Locally

## Tools Used

- **Python / Dash — dashboard framework**
- **Pandas — data loading, cleaning and preprocessing**
- **scikit-learn — TF-IDF + Linear SVM hierarchical baseline classifier**
- **PM4Py — event log construction, Inductive Miner, process mining metrics**
- **Ollama (qwen2.5:7b-instruct) — LLM-based structured extraction**
- **Plotly — interactive process map and Petri net visualisations**
- **Graphviz — process model layout engine (reused for the interactive Petri net)**
- **Sentence embeddings / TF-IDF retrieval — similar-past-query lookup in the live demo**

---

## Overview

Three layers, stacked:

1. **Text Intelligence (RQ1)** — every raw farmer query is sorted into a structured
   `Category` and `QueryType` two ways: a hierarchical TF-IDF + Linear SVM baseline
   classifier, and an LLM-based structured extraction pipeline (Ollama, local). Both are
   evaluated against a hand-labeled gold set; whichever wins feeds the event log below.
2. **Process Mining (RQ2)** — each farmer's sequence of calls (grouped by district + crop)
   is treated as a process-mining "case" and mined with PM4Py's Inductive Miner. This
   surfaces repeat-issue loops: cases where the same type of problem keeps recurring,
   which is a proxy for advice that didn't actually resolve anything the first time.
3. **Semantic Mismatch (RQ3)** — a separate check on whether the answer KCC gave actually
   addresses the question asked, independent of whether the query was correctly
   categorised. A lexical-overlap fallback and a proper NLI (entailment) backend are both
   implemented; see "Honest deviations" below for why the lexical version alone
   undersells this layer.

The dashboard (`dashboard/app.py`) brings all three together: a process explorer with a
live-generated, fully interactive directly-follows graph and Petri net (switchable via
dropdown), crop/case filtering, and a live classifier + retrieval demo where you can type
a farmer-style query and see the predicted category/query type alongside real past KCC
answers to similar questions.

**Note on the numbers below:** the baseline classifier was retuned during development
(`sublinear_tf=True`, `min_df=2`, `C=2.0`, tuned via a validation split carved from the
training data only — the gold set was never touched during tuning). The category/query-type
accuracy figures against the real gold set should be regenerated by re-running
`python src/evaluate_pipelines.py` after retraining, so the numbers reported in the final
write-up reflect the retuned model rather than an earlier run.

## RQ1 — Text Intelligence

```bash
python src/baseline_classifier.py
python src/llm_extraction.py --input data/processed/kcc_clean.csv --model qwen2.5:7b-instruct
python src/llm_extraction.py --consistency-check --model qwen2.5:7b-instruct
python src/evaluate_pipelines.py
```
Compares the baseline classifier and the LLM extraction against
`data/gold/gold_set_final.csv`. Whichever scores higher per-level (accuracy/F1) is the
column used as `--activity-col` in step 2 below — this comparison is the answer to RQ1.

## RQ2 — Process Mining

```bash
python src/event_log.py --input data/processed/kcc_clean.csv --activity-col Category
```
Look at `outputs/summary_by_crop.csv` for loop rate and case duration by crop. This is
where the crop-level repeat-issue finding comes from.

## RQ3 — Semantic Mismatch

```bash
python src/semantic_mismatch.py --backend nli --input data/processed/kcc_clean.csv
```
Cross-reference `outputs/semantic_mismatch.csv`'s flagged rate against the RQ2 loop rate —
do crops/categories with high semantic-mismatch rates also show high loop rates, or are
they catching different problems? That comparison is the answer to RQ3.

## Interactive Process Explorer & Live Classifier Demo

```bash
python dashboard/app.py
```
Open http://127.0.0.1:8050. The process map view defaults to a simple, always-interactive
directly-follows graph; a dropdown switches to the formal Inductive-Miner Petri net, also
fully interactive (built by reusing Graphviz's own layout engine rather than a hand-rolled
one). The live classifier demo at the bottom predicts a category/query type for anything
you type, with a confidence readout for the category stage, plus the five most similar
real past KCC queries and the real answers actually given to them.

## Honest deviations from the original proposal

1. **Label clustering backend**: proposal specified sentence-transformer embeddings for
   merging near-duplicate `QueryType` labels. The dev environment couldn't reach
   HuggingFace, so `data_cleaning.py` defaults to TF-IDF character-similarity clustering
   instead — works everywhere, no external download. Functionally similar for
   near-duplicate *spelling* variants; weaker on true paraphrases.
2. **Semantic mismatch backend**: `semantic_mismatch.py` ships both a TF-IDF lexical
   fallback and the real NLI implementation. The lexical version alone flags the large
   majority of records as mismatches, because good agronomic answers simply don't share
   vocabulary with the problem description — a useful, citable justification for why
   entailment (not surface overlap) is the right tool for RQ3, not a failure of the method.
3. **Case-ID limitation**: `district + crop` is not a real farmer identity — the "loop
   rate" measures repeated issue-categories within a district+crop bucket over time, which
   conflates different farmers who happen to share a crop and district. Worth a
   sensitivity note in the report: a tighter case window (e.g. same month) as a robustness
   check, if time allows.
4. **Gold set size**: sampled and hand-labeled at a scale set by the compressed timeline —
   disclose the exact number used as a scoping decision in the write-up.

## Project Structure & How to Run It Locally

```
src/
  fetch_kcc_data.py         # pulls real data via your data.gov.in API key
  data_cleaning.py          # clean, scope, merge near-duplicate labels
  build_gold_set.py         # sample + merge hand-labeled gold set
  baseline_classifier.py    # RQ1, path A: hierarchical TF-IDF + Linear SVM
  llm_extraction.py         # RQ1, path B: Ollama structured extraction
  evaluate_pipelines.py     # RQ1: compares both paths against the gold set
  retrieval.py              # secondary: similar-past-query retrieval
  event_log.py              # RQ2: PM4Py event log + Inductive Miner + metrics
  semantic_mismatch.py      # RQ3: NLI-based query-answer mismatch detection
dashboard/
  app.py                    # Layer 3: interactive Dash dashboard
data/                        # raw, processed, gold (gitignored except synthetic)
outputs/                     # metrics, csvs, process map exports
models/                      # trained baseline classifier
```

One-time setup:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install graphviz            # or: apt-get install graphviz on Linux
```

Ollama, for the LLM-extraction path:
```bash
brew install ollama              # or download from ollama.com
ollama pull qwen2.5:7b-instruct
```
