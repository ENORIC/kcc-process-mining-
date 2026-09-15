"""
Layer 1, path B: LLM-based structured event extraction via a local Ollama model.

Must be run on a machine with Ollama installed and running
(`ollama serve`, usually automatic once the app is open), since this
sandbox's network policy blocks ollama.com / the Ollama API entirely.

Usage (on your machine):
    ollama pull qwen2.5:7b-instruct
    python src/llm_extraction.py --input data/processed/kcc_clean.csv --model qwen2.5:7b-instruct --n 100

Fixed prompt + low temperature (per proposal) to reduce non-determinism.
--consistency-check re-runs a small subset N times and reports agreement,
so non-determinism is measured rather than assumed away.
"""
import json
import time
import argparse
import re
import requests
import pandas as pd
from collections import Counter

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are extracting a structured event from a farmer helpline call transcript.

Read the QUERY and ANSWER below and output ONLY a single JSON object (no other text) with these exact fields:
- "category": one of ["Plant Protection", "Nutrient Management", "Weather", "Cultural Practices", "Market Information", "Other"]
- "crop": the crop mentioned (single word/phrase, e.g. "Cotton", "Wheat"), or "Unknown" if not mentioned
- "issue": a short (3-6 word) phrase describing the specific problem
- "resolution_signal": one of ["resolved", "unresolved", "unclear"] -- "resolved" if the answer gives a concrete, actionable fix; "unresolved" if the answer is generic/deflects/doesn't address the query; "unclear" if you cannot tell

QUERY: {query}
ANSWER: {answer}

Respond with ONLY the JSON object, nothing else.
"""

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_ollama(prompt: str, model: str, temperature: float = 0.1, timeout: int = 60) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]


def parse_llm_json(raw_text: str) -> dict:
    """Extract and validate the JSON object from a raw LLM response.
    This is unit-testable without a live model -- see test_parse_llm_json()."""
    match = JSON_BLOCK_RE.search(raw_text)
    if not match:
        return {"category": "Other", "crop": "Unknown", "issue": "parse_error", "resolution_signal": "unclear",
                 "_parse_error": True, "_raw": raw_text[:200]}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"category": "Other", "crop": "Unknown", "issue": "parse_error", "resolution_signal": "unclear",
                 "_parse_error": True, "_raw": raw_text[:200]}

    valid_categories = {"Plant Protection", "Nutrient Management", "Weather",
                         "Cultural Practices", "Market Information", "Other"}
    valid_resolution = {"resolved", "unresolved", "unclear"}

    obj.setdefault("category", "Other")
    obj.setdefault("crop", "Unknown")
    obj.setdefault("issue", "unspecified")
    obj.setdefault("resolution_signal", "unclear")

    if obj["category"] not in valid_categories:
        obj["category"] = "Other"
    if obj["resolution_signal"] not in valid_resolution:
        obj["resolution_signal"] = "unclear"
    obj["_parse_error"] = False
    return obj


def extract_record(query: str, answer: str, model: str, temperature: float = 0.1) -> dict:
    prompt = PROMPT_TEMPLATE.format(query=query, answer=answer)
    raw = call_ollama(prompt, model=model, temperature=temperature)
    return parse_llm_json(raw)


def run_extraction(df: pd.DataFrame, model: str, n: int = None, temperature: float = 0.1) -> pd.DataFrame:
    rows = df.head(n) if n else df
    results = []
    for i, row in rows.iterrows():
        try:
            extracted = extract_record(row["QueryText"], row["KccAns"], model=model, temperature=temperature)
        except requests.exceptions.RequestException as e:
            extracted = {"category": "Other", "crop": "Unknown", "issue": "api_error",
                         "resolution_signal": "unclear", "_parse_error": True, "_raw": str(e)}
        extracted["RecordId"] = row.get("RecordId", i)
        results.append(extracted)
        if (len(results) % 25) == 0:
            print(f"Extracted {len(results)}/{len(rows)}...")
    return pd.DataFrame(results)


def consistency_check(df: pd.DataFrame, model: str, n_records: int = 10, n_repeats: int = 3, temperature: float = 0.1):
    """Re-run the same small subset multiple times and report how often the
    extracted category/resolution_signal agree across runs -- reports the
    non-determinism explicitly instead of assuming it away."""
    sample = df.head(n_records)
    agreement_scores = {"category": [], "resolution_signal": []}

    for _, row in sample.iterrows():
        cat_votes, res_votes = [], []
        for _ in range(n_repeats):
            extracted = extract_record(row["QueryText"], row["KccAns"], model=model, temperature=temperature)
            cat_votes.append(extracted["category"])
            res_votes.append(extracted["resolution_signal"])
        cat_agreement = Counter(cat_votes).most_common(1)[0][1] / n_repeats
        res_agreement = Counter(res_votes).most_common(1)[0][1] / n_repeats
        agreement_scores["category"].append(cat_agreement)
        agreement_scores["resolution_signal"].append(res_agreement)

    report = {
        "n_records": n_records,
        "n_repeats": n_repeats,
        "mean_category_agreement": sum(agreement_scores["category"]) / len(agreement_scores["category"]),
        "mean_resolution_agreement": sum(agreement_scores["resolution_signal"]) / len(agreement_scores["resolution_signal"]),
    }
    print("=== LLM consistency check ===")
    print(json.dumps(report, indent=2))
    return report


# ---- offline unit test for the parsing logic (no live Ollama needed) ----
def test_parse_llm_json():
    good = '{"category": "Plant Protection", "crop": "Cotton", "issue": "whitefly attack", "resolution_signal": "resolved"}'
    assert parse_llm_json(good)["category"] == "Plant Protection"

    with_preamble = 'Sure, here is the JSON:\n{"category": "Weather", "crop": "Wheat", "issue": "rain", "resolution_signal": "unresolved"}\nHope that helps!'
    assert parse_llm_json(with_preamble)["crop"] == "Wheat"

    garbage = "I'm not sure how to answer that."
    parsed = parse_llm_json(garbage)
    assert parsed["_parse_error"] is True

    bad_category = '{"category": "Something Weird", "crop": "Rice", "issue": "x", "resolution_signal": "resolved"}'
    assert parse_llm_json(bad_category)["category"] == "Other"

    print("All parse_llm_json tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parser", action="store_true", help="run offline parser unit tests only")
    parser.add_argument("--input", default="data/processed/kcc_clean.csv")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--consistency-check", action="store_true")
    parser.add_argument("--out", default="outputs/llm_extraction.csv")
    args = parser.parse_args()

    if args.test_parser:
        test_parse_llm_json()
    else:
        df = pd.read_csv(args.input)
        if args.consistency_check:
            consistency_check(df, model=args.model)
        else:
            result = run_extraction(df, model=args.model, n=args.n)
            result.to_csv(args.out, index=False)
            print(f"Saved {len(result)} extracted events -> {args.out}")
