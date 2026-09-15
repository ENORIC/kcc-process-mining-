# Semantic-Aware LLM Process Mining Advisory Workflow (KCC)

## STATUS as of Sept 15 -- real results already in hand, read this first

Real data: Kerala, 2024, downloaded via data.gov.in's Data tab (no API needed).
26,601 raw records -> 6,570 after cleaning (dropped admin queries like
"Government Schemes"/"Training", dropped Malayalam-script answers -- see
"Honest deviations" below for why).

**Already done, with real numbers:**
- Baseline classifier (RQ1, path A): Category accuracy 86.3% (F1 macro 0.70),
  QueryType accuracy 75.8% (F1 macro 0.29 -- expected, 42 imbalanced classes).
  See `outputs/baseline_classifier_metrics.json`.
- Event log + process mining (RQ2): 711 cases (district+crop), mean loop rate
  0.28, 343/711 cases show repeat-issue looping. **Coconut (0.86) and Banana
  (0.85) have the highest loop rates** among crops with meaningful volume --
  this is a real, reportable finding. See `outputs/summary_by_crop.csv`,
  `outputs/process_model.png` (filtered to top-8 activities + top-15 variants
  for legibility -- the unfiltered model is an unreadable spaghetti graph,
  worth mentioning honestly rather than claiming Inductive Miner is
  automatically readable at this activity count).
- Semantic mismatch, lexical fallback: flags 94.8% of records as mismatches
  -- confirms (again, now on real data) that surface lexical overlap can't
  tell good agronomic answers from bad ones, which is the actual
  justification for why RQ3 needs real NLI, not a failure.
- Dashboard: boots and renders correctly against real outputs. Live query
  demo tested: "coconut tree leaves turning yellow, what fertilizer should
  I use" -> correctly predicts Category=Plantation Crops, QueryType=Nutrient
  Management.
- Gold set sampled: `data/gold/gold_set_to_label.csv`, 60 records, pre-filled
  with existing labels -- **you need to hand-verify these** (see step 3).

**Still needs you, specifically:**
1. Hand-label `data/gold/gold_set_to_label.csv` (60 records) -- read each
   query+answer, correct `Category_verified`/`QueryType_verified` where
   wrong, fill in `resolution_signal_verified`. Then run
   `python src/build_gold_set.py merge`.
2. Run the real LLM extraction via Ollama (RQ1, path B) -- this has to run
   on your machine, model's already pulled:
   ```
   python src/llm_extraction.py --input data/processed/kcc_clean.csv --model qwen2.5:7b-instruct --n 200
   python src/llm_extraction.py --consistency-check --model qwen2.5:7b-instruct
   ```
   Compare accuracy against `data/gold/gold_set_final.csv` -- this comparison
   IS your RQ1 answer.
3. (Optional, if time) Real NLI semantic mismatch instead of the lexical
   fallback: `python src/semantic_mismatch.py --backend nli`
4. Write up the report using the numbers above plus whatever your LLM
   extraction run adds.


Built under a hard deadline (submission 17th) — this README is the exact
runbook to go from zero to a working demo on YOUR machine. Everything here
was developed and unit-tested against a synthetic stand-in dataset
(`data/raw/kcc_synthetic.csv`, schema-matched to the real KCC columns) in an
environment with no internet access to data.gov.in / Ollama / HuggingFace.
The real run happens on your machine, where all three are reachable.

## 0. One-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install graphviz            # or: apt-get install graphviz on Linux
```

Put your data.gov.in API key in a local `.env` file (never commit this):
```
KCC_API_KEY=your-key-here
```

Install Ollama and pull a model:
```bash
brew install ollama              # or download from ollama.com
ollama pull qwen2.5:7b-instruct  # or a smaller/faster model if this is too slow on your machine
```

## 1. Get the real data

```bash
python src/fetch_kcc_data.py --test
```
Run this FIRST and check the printed field names against `REQUIRED_COLUMNS`
in `src/data_cleaning.py` (line ~15) — the real API may use slightly
different column names than assumed. Fix the mapping there if needed, then:

```bash
python src/fetch_kcc_data.py --limit 20000 --out data/raw/kcc_real.csv
```

If the API turns out to be slow/limited, the Kaggle dataset
"Farmers Call Query (KCC) Data" (daskoushik/farmers-call-query-data-qa) is
the same underlying government data, pre-packaged — a fine substitute,
just note it in the report as the actual data source used.

## 2. Clean + scope the data

```bash
python src/data_cleaning.py --raw data/raw/kcc_real.csv --out data/processed/kcc_clean.csv \
    --states "RAJASTHAN" "UTTAR PRADESH" --crops "Cotton" "Wheat" "Mustard"
```
Adjust `--states`/`--crops` to whatever you scoped in the proposal. Check the
printed record counts at each stage — if scoping leaves too few records
(< a few hundred), widen it.

## 3. Build the gold-labeled evaluation set

```bash
python src/build_gold_set.py sample --n 100
```
Open `data/gold/gold_set_to_label.csv`, correct the `*_verified` columns
where the pre-filled (noisy) label is wrong, and fill in
`resolution_signal_verified` (resolved / unresolved / unclear) for each row
by actually reading the query+answer. Then:
```bash
python src/build_gold_set.py merge
```

## 4. Layer 1 — Text Intelligence (RQ1)

Baseline classifier:
```bash
python src/baseline_classifier.py
```
LLM extraction (needs Ollama running):
```bash
python src/llm_extraction.py --input data/processed/kcc_clean.csv --model qwen2.5:7b-instruct
python src/llm_extraction.py --consistency-check --model qwen2.5:7b-instruct
```
Compare both against `data/gold/gold_set_final.csv` — whichever scores
higher (per-level accuracy/F1) is what feeds the event log in step 5. This
comparison IS the answer to RQ1 — write down both numbers.

Retrieval (secondary feature):
```bash
python src/retrieval.py --query "some example farmer query" --backend sbert
```

## 5. Layer 2 — Process Mining (RQ2, RQ3)

```bash
python src/event_log.py --input data/processed/kcc_clean.csv --activity-col Category
python src/semantic_mismatch.py --backend nli --input data/processed/kcc_clean.csv
```
Use whichever column (Category from the classifier, or the LLM's extracted
category) won the Layer 1 comparison as `--activity-col`.

Look at `outputs/summary_by_crop.csv` (loop rate + duration by crop) for
RQ2, and cross-reference `outputs/semantic_mismatch.csv`'s flagged rate
against the loop rate — do crops/categories with high semantic-mismatch
rates also show high loop rates, or are they catching different problems?
That comparison is the answer to RQ3.

## 6. Layer 3 — Dashboard

```bash
python dashboard/app.py
```
Open http://127.0.0.1:8050

## Honest deviations from the original proposal (disclose these in the report)

1. **Label clustering backend**: proposal specified sentence-transformer
   embeddings for merging near-duplicate QueryType labels. The dev
   environment couldn't reach HuggingFace, so `data_cleaning.py` defaults to
   TF-IDF character-similarity clustering instead — works everywhere, no
   external download. Functionally similar for near-duplicate *spelling*
   variants; weaker on true paraphrases. Swap in the sentence-transformers
   version if time allows (see `retrieval.py`'s `SbertFaissRetriever` for
   the pattern).
2. **Semantic mismatch backend**: same story — `semantic_mismatch.py` has a
   TF-IDF lexical fallback and the real NLI implementation. The lexical one
   was tested on synthetic data and flagged ~100% of records as mismatches,
   because good agronomic answers don't lexically overlap with the problem
   description at all — this is actually a good, citable justification for
   why NLI/entailment (not surface overlap) is the right tool for RQ3, not
   a failure.
3. **Case-ID limitation**: as already flagged in the proposal, `district +
   crop` is not a real farmer identity — the "loop rate" in
   `event_log.py` measures repeated issue-categories within a
   district+crop bucket over time, which conflates different farmers with
   the same crop in the same district. Worth a sensitivity note in the
   report: try a tighter case window (e.g. same month) as a robustness
   check if time allows.
4. **Sample sizes**: gold set defaults to 100 records (proposal said
   300-500) given the compressed timeline — disclose as a scoping decision.

## Project structure

```
src/
  fetch_kcc_data.py        # pulls real data via your API key (run on your machine)
  generate_synthetic_data.py  # dev-only stand-in, not part of the submission
  data_cleaning.py         # clean, scope, merge near-duplicate labels
  build_gold_set.py        # sample + merge hand-labeled gold set
  baseline_classifier.py   # Layer 1, path A: hierarchical TF-IDF+SVM
  llm_extraction.py        # Layer 1, path B: Ollama structured extraction
  retrieval.py             # secondary: similar-past-query retrieval
  event_log.py             # Layer 2: PM4Py event log + Inductive Miner + metrics
  semantic_mismatch.py     # RQ3: NLI-based query-answer mismatch detection
dashboard/
  app.py                   # Layer 3: Dash dashboard
data/                       # raw, processed, gold (gitignored except synthetic)
outputs/                    # metrics, csvs, process map image
models/                     # trained baseline classifier
```
