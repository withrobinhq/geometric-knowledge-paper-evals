#!/usr/bin/env python3
"""Normalize two pilot sheets, an optional answer key, and match metadata."""

import argparse
import csv
import re
from pathlib import Path


INPUT_COLUMNS = ["row_type", "item_id", "prompt", "D1", "D2", "D3", "D4", "grade", "note"]
OUTPUT_COLUMNS = [
    "record_type",
    "annotator_id",
    "blind_to_peer",
    "blind_to_key",
    "item_id",
    "query_id",
    "rank",
    "claim_id",
    "prompt",
    "dimension_id",
    "field_name",
    "value",
    "grade",
    "note",
    "match_dimension_a",
    "match_dimension_b",
    "axis_match",
    "direction",
    "match_filled_at",
    "match_note",
]


def read_sheet(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(INPUT_COLUMNS) - set(rows[0] if rows else INPUT_COLUMNS)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return [row for row in rows if (row.get("row_type") or "").strip() != "HELP"]


def scrub(text, names):
    value = text or ""
    for name in names:
        if name:
            value = re.sub(rf"\b{re.escape(name)}\b", "annotator", value, flags=re.IGNORECASE)
    return value


def split_relevance_id(item_id):
    parts = item_id.split("|")
    return parts if len(parts) == 3 else ("", "", "")


def sheet_records(path, annotator_id, blind_to_peer, blind_to_key, names):
    rows = read_sheet(path)
    authored = set()
    for row in rows:
        if row["row_type"] == "DIM_FIELD" and row["item_id"] == "name":
            for dim in ("D1", "D2", "D3", "D4"):
                if (row.get(dim) or "").strip() not in ("", "-"):
                    authored.add(dim)

    records = []
    base = {
        "annotator_id": annotator_id,
        "blind_to_peer": blind_to_peer,
        "blind_to_key": blind_to_key,
    }
    for row in rows:
        kind, item_id = row["row_type"].strip(), row["item_id"].strip()
        prompt = scrub(row.get("prompt", ""), names)
        note = scrub(row.get("note", ""), names)
        if kind == "DIM_FIELD":
            for dim in sorted(authored):
                records.append({
                    **base,
                    "record_type": "dimension_field",
                    "item_id": item_id,
                    "prompt": prompt,
                    "dimension_id": dim,
                    "field_name": item_id,
                    "value": scrub(row.get(dim, ""), names),
                    "note": note,
                })
        elif kind == "POS":
            for dim in sorted(authored):
                records.append({
                    **base,
                    "record_type": "position",
                    "item_id": item_id,
                    "claim_id": item_id,
                    "prompt": prompt,
                    "dimension_id": dim,
                    "value": scrub(row.get(dim, ""), names),
                    "note": note,
                })
        elif kind == "REL":
            query_id, rank, claim_id = split_relevance_id(item_id)
            records.append({
                **base,
                "record_type": "relevance",
                "item_id": item_id,
                "query_id": query_id,
                "rank": rank,
                "claim_id": claim_id,
                "prompt": prompt,
                "grade": row.get("grade", ""),
                "note": note,
            })
        elif kind == "TIME":
            records.append({
                **base,
                "record_type": "time",
                "item_id": item_id,
                "field_name": item_id,
                "value": note,
            })
    return records


def merge(
    sheet_a,
    sheet_b,
    output,
    match=None,
    key=None,
    a_blind_to_peer="true",
    a_blind_to_key="true",
    b_blind_to_peer="true",
    b_blind_to_key="true",
    names=(),
):
    merged = sheet_records(
        sheet_a, "A", a_blind_to_peer, a_blind_to_key, names
    ) + sheet_records(
        sheet_b, "B", b_blind_to_peer, b_blind_to_key, names
    )
    if key:
        merged += sheet_records(key, "KEY", "", "", names)

    if match:
        with open(match, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("row_type") or "").strip() != "MATCH":
                    continue
                merged.append(
                    {
                        "record_type": "match",
                        "match_dimension_a": row.get("dim_a", ""),
                        "match_dimension_b": row.get("dim_b", ""),
                        "axis_match": row.get("same_axis", ""),
                        "direction": row.get("direction", ""),
                        "match_filled_at": row.get("filled_at", ""),
                        "match_note": scrub(row.get("note", ""), names),
                    }
                )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sheet_a")
    parser.add_argument("sheet_b")
    parser.add_argument("output")
    parser.add_argument("--match")
    parser.add_argument("--key")
    parser.add_argument("--a-blind-to-peer", default="true")
    parser.add_argument("--a-blind-to-key", default="true")
    parser.add_argument("--b-blind-to-peer", default="true")
    parser.add_argument("--b-blind-to-key", default="true")
    parser.add_argument("--redact-name", action="append", default=[])
    args = parser.parse_args()
    merge(
        args.sheet_a,
        args.sheet_b,
        args.output,
        match=args.match,
        key=args.key,
        a_blind_to_peer=args.a_blind_to_peer,
        a_blind_to_key=args.a_blind_to_key,
        b_blind_to_peer=args.b_blind_to_peer,
        b_blind_to_key=args.b_blind_to_key,
        names=args.redact_name,
    )


if __name__ == "__main__":
    main()
