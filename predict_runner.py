"""
predict_runner.py

Runs predict_price() and prints the result as JSON to stdout.
Invoked as a subprocess from api.py so XGBoost never shares a process
with PyTorch (their bundled libomp libraries collide and crash Python).

Usage:
    python predict_runner.py AAPL 2026-06-01
"""

import json
import sys

from predict import predict_price


def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "usage: predict_runner.py TICKER YYYY-MM-DD"}))
        sys.exit(2)

    ticker = sys.argv[1]
    target_date = sys.argv[2]

    try:
        result = predict_price(ticker, target_date)
        print(json.dumps(result))
    except ValueError as e:
        print(json.dumps({"error": str(e), "kind": "value"}))
        sys.exit(1)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e), "kind": "file"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e), "kind": "other"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
