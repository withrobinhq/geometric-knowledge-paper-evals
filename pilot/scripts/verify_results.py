#!/usr/bin/env python3
"""Verify that scored CSV fragments exactly reproduce pilot_results.csv."""

import argparse
import csv
import sys


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical(rows):
    return sorted(
        (tuple(sorted(row.items())) for row in rows),
        key=repr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("actual", nargs="+")
    args = parser.parse_args()
    expected = read_rows(args.expected)
    actual = []
    for path in args.actual:
        actual.extend(read_rows(path))
    if canonical(expected) != canonical(actual):
        print(f"FAIL: expected {len(expected)} rows, got {len(actual)}", file=sys.stderr)
        return 1
    print(f"PASS: {len(actual)} pilot result rows reproduced exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
