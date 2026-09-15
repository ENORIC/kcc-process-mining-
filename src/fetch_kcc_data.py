"""
Pulls the real KCC dataset from data.gov.in using your API key.

Run this on YOUR machine (not in a sandbox with restricted network) -- the
data.gov.in resource API needs to be reachable directly.

Setup (once):
    export KCC_API_KEY="your-key-here"
    # or put KCC_API_KEY=your-key-here in a .env file in the project root
    pip install requests python-dotenv pandas

Usage:
    python src/fetch_kcc_data.py --test              # fetch 5 records, print raw field names
    python src/fetch_kcc_data.py --limit 20000        # full pull, paginated, saved to CSV
    python src/fetch_kcc_data.py --limit 20000 --state "RAJASTHAN" --state "UTTAR PRADESH"

NEVER commit your real API key. It should only ever live in your shell env
or an untracked .env file (see .gitignore).
"""
import os
import sys
import time
import argparse
import requests
import pandas as pd

RESOURCE_ID = "cef25fe2-9231-4128-8aec-2c948fedd43f"  # Kisan Call Centre (KCC) transcripts
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


def get_api_key():
    key = os.environ.get("KCC_API_KEY")
    if not key:
        # try loading from a local .env file without requiring python-dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("KCC_API_KEY"):
                        key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not key:
        print("ERROR: KCC_API_KEY not set. Export it or put it in a .env file (see this file's docstring).")
        sys.exit(1)
    return key


def fetch_page(api_key, limit=1000, offset=0, filters=None):
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
        "offset": offset,
    }
    if filters:
        for field, value in filters.items():
            params[f"filters[{field}]"] = value
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="fetch just 5 records and print field names")
    parser.add_argument("--limit", type=int, default=20000, help="total records to fetch (paginated)")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--state", action="append", default=None, help="filter field value, repeatable")
    parser.add_argument("--out", default="data/raw/kcc_real.csv")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.test:
        data = fetch_page(api_key, limit=5, offset=0)
        print("Top-level keys:", list(data.keys()))
        records = data.get("records", [])
        print(f"Got {len(records)} sample records")
        if records:
            print("Field names in each record:", list(records[0].keys()))
            print("\nFirst record:\n", records[0])
        return

    all_records = []
    offset = 0
    while len(all_records) < args.limit:
        page = min(args.page_size, args.limit - len(all_records))
        data = fetch_page(api_key, limit=page, offset=offset)
        records = data.get("records", [])
        if not records:
            print("No more records returned, stopping.")
            break
        all_records.extend(records)
        offset += page
        print(f"Fetched {len(all_records)} records so far...")
        time.sleep(0.3)  # be polite to the API

    df = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} records -> {args.out}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
