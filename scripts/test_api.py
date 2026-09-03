"""Simple client to validate POST /predict after starting the local API."""
from __future__ import annotations

import json
import sys

import requests

URL = "http://127.0.0.1:8000/predict"
SAMPLE_TRANSACTION = {
    "Time": 10.2,
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.53,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": 0.24,
    "V8": 0.10,
    "V9": 0.36,
    "V10": 0.09,
    "V11": -0.55,
    "V12": -0.62,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.47,
    "V16": -0.47,
    "V17": 0.20,
    "V18": 0.03,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.02,
    "V22": 0.28,
    "V23": -0.11,
    "V24": 0.07,
    "V25": 0.13,
    "V26": -0.19,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62,
}


def main() -> int:
    response = requests.post(URL, json=SAMPLE_TRANSACTION, timeout=15)
    print("status_code=", response.status_code)
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(response.text)
    return 0 if response.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
