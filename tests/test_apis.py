"""
Quick test script to verify API keys and data sources are working.
Run from project root: python -m tests.test_apis
"""

import requests


# ============================================================
# TEST 1: FRED API
# ============================================================
def test_fred():
    print("=" * 60)
    print("TEST 1: FRED API")
    print("=" * 60)

    api_key = "018111c521487e5f3c15f637a23dd632"
    series_ids = {
        "FEDFUNDS": "Federal Funds Rate",
        "DGS10": "10-Year Treasury Rate",
        "M2SL": "M2 Money Supply",
    }

    for series_id, name in series_ids.items():
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "observations" in data and len(data["observations"]) > 0:
                obs = data["observations"][0]
                print(f"  [PASS] {name} ({series_id}): {obs['value']} (date: {obs['date']})")
            else:
                print(f"  [FAIL] {name} ({series_id}): No data returned")
        except Exception as e:
            print(f"  [FAIL] {name} ({series_id}): ERROR -- {e}")

    print()


# ============================================================
# TEST 2: Reddit JSON Scraping (no API key needed)
# ============================================================
def test_reddit_json():
    print("=" * 60)
    print("TEST 2: Reddit JSON Scraping (no API key)")
    print("=" * 60)

    subreddits = ["solana", "dogecoin", "CryptoCurrency"]
    headers = {"User-Agent": "CPT/1.0 (Crypto Price Tracker)"}

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit=3"

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            posts = data.get("data", {}).get("children", [])
            if posts:
                print(f"  [PASS] r/{sub} -- got {len(posts)} posts:")
                for post in posts:
                    title = post["data"]["title"][:70]
                    score = post["data"]["score"]
                    print(f"      [{score:>5} pts] {title}")
            else:
                print(f"  [FAIL] r/{sub} -- no posts returned")
        except Exception as e:
            print(f"  [FAIL] r/{sub} -- ERROR: {e}")

    print()


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == "__main__":
    print("\nCPT API Key Verification\n")
    test_fred()
    test_reddit_json()
    print("All tests completed.")
