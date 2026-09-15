"""
Generates a synthetic dataset that mirrors the real KCC (Kisan Call Centre)
transcript schema, so the rest of the pipeline can be built and tested
BEFORE the real dataset is downloaded from kcc-chakshu.icar.gov.in / Kaggle.

Real KCC columns (per data.gov.in / AIKosh / literature, e.g. Godara et al. 2024):
    StateName, DistrictName, BlockName, Season,
    Sector, Category, Crop, QueryType,
    QueryText, KccAns, CreatedOn

Once the real CSV is downloaded, point data_cleaning.py at it instead of
this synthetic file -- the schema and downstream code are designed to be
a drop-in match.

Deliberately injects:
  - repeat-callers: same (district, crop) asking a near-duplicate question
    within a short window -> should show up as loop behaviour in Layer 2.
  - a fraction of "mismatched" answers (answer doesn't address the query)
    -> should be caught by the NLI mismatch detector in Layer 2.
  - noisy/inconsistent QueryType labels for the same underlying issue
    -> tests the near-duplicate label clustering step.
"""
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

STATES = {
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Bikaner"],
    "Uttar Pradesh": ["Lucknow", "Kanpur Nagar", "Meerut", "Varanasi"],
}

CROPS = ["Cotton", "Wheat", "Mustard", "Bajra", "Chickpea"]

SECTOR = "Agriculture"  # KCC is ~all Agriculture/Horticulture/Animal Husbandry; kept simple here

CATEGORY_QUERYTYPE = {
    "Plant Protection": ["Pest attack", "Pest Management", "Insect infestation", "Disease control", "Fungal disease"],
    "Nutrient Management": ["Fertilizer dose", "Fertiliser use", "Micronutrient deficiency", "Soil health"],
    "Weather": ["Rainfall query", "Weather advisory", "Irrigation timing"],
    "Cultural Practices": ["Sowing time", "Seed rate", "Variety selection", "Spacing query"],
    "Market Information": ["Price query", "Mandi rate", "Selling advice"],
}

ISSUE_TEMPLATES = {
    "Plant Protection": [
        "whitefly attack on {crop}", "bollworm infestation in {crop}", "leaf curl disease in {crop}",
        "aphid attack on {crop}", "powdery mildew on {crop}",
    ],
    "Nutrient Management": [
        "yellowing of {crop} leaves, nutrient deficiency", "poor growth of {crop}, needs fertilizer advice",
        "{crop} showing stunted growth",
    ],
    "Weather": [
        "unseasonal rain affecting {crop}", "when to irrigate {crop} this week", "drought stress on {crop}",
    ],
    "Cultural Practices": [
        "best sowing time for {crop} this season", "which {crop} variety to use", "seed rate for {crop}",
    ],
    "Market Information": [
        "current mandi price for {crop}", "where to sell {crop} for best price",
    ],
}

GOOD_ANSWER_TEMPLATES = {
    "Plant Protection": "Spray {chemical} @ {dose} per litre of water in the evening, repeat after 10 days if needed. Avoid spraying during flowering.",
    "Nutrient Management": "Apply {chemical} @ {dose} per acre as a foliar spray, and ensure balanced NPK application based on soil test.",
    "Weather": "Based on the current forecast, irrigate {crop} within the next 2-3 days and avoid standing water.",
    "Cultural Practices": "For {crop} in this region and season, sow using variety RH-30 at a seed rate of 5kg/acre with 30cm row spacing.",
    "Market Information": "Current modal price for {crop} at the nearest mandi is around Rs. 2200-2600 per quintal.",
}

MISMATCH_ANSWER_TEMPLATES = [
    "Please contact your nearest Krishi Vigyan Kendra for general information.",
    "The weather this week is expected to be sunny with light winds.",
    "Kindly visit the block agriculture office for further assistance.",
    "Thank you for calling Kisan Call Centre, your query has been noted.",
]

CHEMICALS = ["Imidacloprid 17.8% SL", "Chlorpyrifos 20% EC", "Urea", "DAP", "NPK 19:19:19", "Mancozeb 75% WP"]
DOSES = ["1ml", "2ml", "1.5g", "2kg", "500g"]


def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def make_query_answer(category, crop, mismatch=False):
    issue_template = random.choice(ISSUE_TEMPLATES[category])
    query = f"Farmer reports {issue_template.format(crop=crop)}. What should be done?"
    if mismatch:
        answer = random.choice(MISMATCH_ANSWER_TEMPLATES)
    else:
        answer = GOOD_ANSWER_TEMPLATES[category].format(
            crop=crop, chemical=random.choice(CHEMICALS), dose=random.choice(DOSES)
        )
    return query, answer


def generate(n_base_cases=400, repeat_rate=0.18, mismatch_rate=0.15, out_path="data/raw/kcc_synthetic.csv"):
    rows = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    record_id = 1

    for _ in range(n_base_cases):
        state = random.choice(list(STATES.keys()))
        district = random.choice(STATES[state])
        crop = random.choice(CROPS)
        category = random.choice(list(CATEGORY_QUERYTYPE.keys()))
        querytype_clean = random.choice(CATEGORY_QUERYTYPE[category])
        # inject label noise: sometimes a near-duplicate but differently-worded QueryType
        querytype_noisy = querytype_clean if random.random() > 0.3 else querytype_clean.lower() + " issue"

        is_mismatch = random.random() < mismatch_rate
        query, answer = make_query_answer(category, crop, mismatch=is_mismatch)
        call_date = random_date(start_date, end_date)

        rows.append({
            "RecordId": record_id,
            "StateName": state,
            "DistrictName": district,
            "BlockName": f"{district} Block-{random.randint(1,3)}",
            "Season": random.choice(["Kharif", "Rabi", "Zaid"]),
            "Sector": SECTOR,
            "Category": category,
            "Crop": crop,
            "QueryType": querytype_noisy,
            "QueryText": query,
            "KccAns": answer,
            "CreatedOn": call_date.strftime("%Y-%m-%d"),
        })
        record_id += 1

        # simulate a repeat call: same district+crop, same underlying issue,
        # a few days later, IF the first answer was a mismatch (unresolved) -> loop
        if is_mismatch and random.random() < (repeat_rate / mismatch_rate if mismatch_rate else 0):
            follow_up_date = call_date + timedelta(days=random.randint(2, 12))
            if follow_up_date <= end_date:
                # second call still unresolved sometimes, resolved other times
                still_mismatch = random.random() < 0.55
                q2, a2 = make_query_answer(category, crop, mismatch=still_mismatch)
                rows.append({
                    "RecordId": record_id,
                    "StateName": state,
                    "DistrictName": district,
                    "BlockName": f"{district} Block-{random.randint(1,3)}",
                    "Season": random.choice(["Kharif", "Rabi", "Zaid"]),
                    "Sector": SECTOR,
                    "Category": category,
                    "Crop": crop,
                    "QueryType": querytype_noisy,
                    "QueryText": q2,
                    "KccAns": a2,
                    "CreatedOn": follow_up_date.strftime("%Y-%m-%d"),
                })
                record_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df["RecordId"] = range(1, len(df) + 1)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic records -> {out_path}")
    print(df["Category"].value_counts())
    print(f"Mismatch-flagged (ground truth, for validation only): "
          f"{sum(1 for r in rows if r['KccAns'] in MISMATCH_ANSWER_TEMPLATES)}")
    return df


if __name__ == "__main__":
    generate()
