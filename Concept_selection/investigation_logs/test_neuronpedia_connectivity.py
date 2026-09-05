"""
Neuronpedia API connectivity test.

Confirms we can actually fetch real feature data for our LOCKED target:
model=gemma-3-1b-it, SAE source=14-gemmascope-2-res-16k (layer 14, the
"nearest" resolution of the 50-60% relative depth band for this model's
26 layers, per the Context Pack).

Uses NEURONPEDIA_API_KEY from the environment -- never hardcoded, never
printed. This is a read-only test: one GET request, no writes, no
bookmarks, no uploads.
"""

import os
import sys

import requests

NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"
TARGET_MODEL_ID = "gemma-3-1b-it"
TARGET_SOURCE_ID = "14-gemmascope-2-res-16k"
TARGET_FEATURE_INDEX = "0"  # arbitrary first feature, just to confirm the endpoint responds


def get_api_key() -> str:
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NEURONPEDIA_API_KEY is not set in the environment. "
            "Run 'setx NEURONPEDIA_API_KEY \"your_key\"' and open a fresh terminal."
        )
    return api_key


def fetch_one_feature(model_id: str, source_id: str, feature_index: str, api_key: str) -> dict:
    url = f"{NEURONPEDIA_BASE_URL}/feature/{model_id}/{source_id}/{feature_index}"
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print(f"Testing Neuronpedia API access for {TARGET_MODEL_ID}/{TARGET_SOURCE_ID}...")

    try:
        key = get_api_key()
    except EnvironmentError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    try:
        feature_data = fetch_one_feature(TARGET_MODEL_ID, TARGET_SOURCE_ID, TARGET_FEATURE_INDEX, key)
    except requests.exceptions.HTTPError as e:
        print(f"FAILED: HTTP error - {e}")
        print(f"Response body: {e.response.text[:500]}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"FAILED: Request error - {e}")
        sys.exit(1)

    print("SUCCESS: Received a real response from Neuronpedia.")
    print(f"  modelId: {feature_data.get('modelId')}")
    print(f"  layer/source: {feature_data.get('layer')}")
    print(f"  index: {feature_data.get('index')}")
    # Print a short preview of any auto-interpretation text, if present,
    # just to visually confirm this is real feature data, not an empty stub.
    explanations = feature_data.get("explanations", [])
    if explanations:
        print(f"  first explanation preview: {explanations[0].get('description', '')[:100]!r}")
    else:
        print("  (no explanations field found in response -- worth inspecting the full JSON)")
