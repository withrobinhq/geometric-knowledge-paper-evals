"""Score one merged anonymous pilot CSV: Dimension agreement, relevance agreement, per-item SDs.

Usage:
  score.py merged.csv [--compare A:B] [--dataset name] [--csv out.csv]
  score.py --selftest
"""
import csv
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import krippendorff

DIMS = ["D1", "D2", "D3", "D4"]
COLUMNS = ["row_type", "item_id", "prompt", *DIMS, "grade", "note"]
NR = "NR"


# ---------- parsing ----------

def parse_value_space(text):
    """Return ('ordinal', [levels]) or ('scalar', None); raise ValueError otherwise."""
    t = (text or "").strip()
    kind, _, rest = t.partition(":")
    kind = kind.strip().lower()
    if kind == "ordinal":
        levels = [x.strip().lower() for x in rest.split("<") if x.strip()]
        if not 3 <= len(levels) <= 5 or len(set(levels)) != len(levels):
            raise ValueError("ordinal scale needs 3 to 5 distinct levels written low < ... < high")
        return "ordinal", levels
    if kind == "scalar":
        return "scalar", None
    if kind in ("nominal", "categorical", "categories"):
        raise ValueError("unordered categories are not admitted (Section 5.1)")
    raise ValueError(f"value_space must start with 'ordinal:' or 'scalar:', got {t!r}")


def normalized(raw, space):
    """Map a recorded position to [0, 1] (the normalizer nu_d). NR -> None."""
    v = (raw or "").strip()
    if not v:
        raise ValueError("blank position (use NR when the Claim does not concern the Dimension)")
    if v.upper() == NR:
        return None
    kind, levels = space
    if kind == "ordinal":
        if v.lower() in levels:
            return levels.index(v.lower()) / (len(levels) - 1)
        if v.isdigit() and 1 <= int(v) <= len(levels):
            return (int(v) - 1) / (len(levels) - 1)
        raise ValueError(f"{v!r} is not one of {levels}")
    x = float(v)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{v!r} is outside [0, 1]")
    return x


def contract_problems(fields, space):
    problems = []
    kind = space[0]
    comp = fields.get("comparison", "").lower()
    if kind == "ordinal" and "rank" not in comp:
        problems.append("comparison should be rank distance for an ordinal scale")
    if kind == "scalar" and "absolute" not in comp:
        problems.append("comparison should be absolute difference for a scalar")
    protos = [p for p in fields.get("prototypes", "").split(";") if p.strip()]
    if not 2 <= len(protos) <= 5:
        problems.append(f"needs 2 to 5 prototypes, found {len(protos)}")
    for p in protos:
        _, sep, val = p.rpartition("=")
        try:
            if not sep or normalized(val, space) is None:
                raise ValueError
        except ValueError:
            problems.append(f"prototype without a valid value: {p.strip()[:40]!r}")
    for f in ("name", "definition", "provenance", "version"):
        if not fields.get(f, "").strip():
            problems.append(f"missing {f}")
    return problems


def read_sheet(path):
    sheet = {"path": str(path), "dims": {}, "positions": {}, "grades": {}, "minutes": {}, "errors": []}
    fields = {d: {} for d in DIMS}
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        kind = (r.get("row_type") or "").strip()
        if kind == "DIM_FIELD":
            for d in DIMS:
                fields[d][r["item_id"].strip()] = (r.get(d) or "").strip()
    for d in DIMS:
        if not fields[d].get("name") or fields[d]["name"] == "-":
            continue
        try:
            space = parse_value_space(fields[d].get("value_space"))
        except ValueError as e:
            sheet["errors"].append(f"{d}: {e}")
            continue
        sheet["dims"][d] = {"fields": fields[d], "space": space,
                            "problems": contract_problems(fields[d], space)}
    for r in rows:
        kind, item = (r.get("row_type") or "").strip(), (r.get("item_id") or "").strip()
        if kind == "POS":
            for d, dim in sheet["dims"].items():
                try:
                    sheet["positions"].setdefault(item, {})[d] = normalized(r.get(d), dim["space"])
                except ValueError as e:
                    sheet["errors"].append(f"{item} {d}: {e}")
        elif kind == "REL":
            g = (r.get("grade") or "").strip()
            if g in ("0", "1", "2", "3"):
                sheet["grades"][item] = int(g)
            else:
                sheet["errors"].append(f"{item}: grade must be 0, 1, 2 or 3, got {g!r}")
        elif kind == "TIME":
            m = (r.get("note") or "").strip()
            if m.isdigit():
                sheet["minutes"][item] = int(m)
    return sheet


def read_match(path):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("row_type") or "").strip() != "MATCH":
                continue
            same = (r.get("same_axis") or "").strip().lower()
            if same in ("yes", "partial"):
                pairs.append({"a": r["dim_a"].strip(), "b": r["dim_b"].strip(), "same_axis": same,
                              "reversed": (r.get("direction") or "").strip().lower() == "reversed"})
    return pairs


def auto_match(a, b):
    key = lambda f: (f["name"].strip().lower(), f["definition"].strip().lower())  # noqa: E731
    pairs = []
    for da, x in a["dims"].items():
        for db, y in b["dims"].items():
            if key(x["fields"]) == key(y["fields"]):
                pairs.append({"a": da, "b": db, "same_axis": "yes", "reversed": False})
    return pairs


# ---------- statistics ----------

def alpha(x, y, level, value_domain=None):
    """value_domain lists every level of an ordinal scale, so unused levels still count."""
    data = np.array([x, y], dtype=float)
    both = ~np.isnan(data).any(axis=0)
    if both.sum() < 2 or len(np.unique(data[:, both])) < 2:
        return None
    return float(krippendorff.alpha(reliability_data=data, level_of_measurement=level,
                                    value_domain=value_domain))


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def cohen_kappa(x, y):
    cats = sorted(set(x) | set(y))
    n = len(x)
    if n == 0:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def item_sd(x, y):
    """Per-item SD with two judgments is |a - b| / sqrt(2) (sample SD, n = 2)."""
    sds = [abs(a - b) / math.sqrt(2) for a, b in zip(x, y)]
    if not sds:
        return {"n": 0, "mean_item_sd": None, "pooled_item_sd": None}
    return {"n": len(sds), "mean_item_sd": float(np.mean(sds)),
            "pooled_item_sd": float(math.sqrt(np.mean(np.square(sds))))}


def ndcg(grades_in_rank_order):
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades_in_rank_order))
    ideal = sum((2 ** g - 1) / math.log2(i + 2)
                for i, g in enumerate(sorted(grades_in_rank_order, reverse=True)))
    return None if ideal == 0 else dcg / ideal


# ---------- scoring ----------

def score_pair(a, b, pairs):
    claims = sorted(set(a["positions"]) & set(b["positions"]))
    out = []
    for p in pairs:
        if p["a"] not in a["dims"] or p["b"] not in b["dims"]:
            out.append({**p, "error": "dimension not authored in one of the sheets"})
            continue
        sa, sb = a["dims"][p["a"]]["space"], b["dims"][p["b"]]["space"]
        xa = [a["positions"][c].get(p["a"]) for c in claims]
        xb = [b["positions"][c].get(p["b"]) for c in claims]
        if p["reversed"]:
            xb = [None if v is None else 1 - v for v in xb]
        same_ordinal = sa[0] == sb[0] == "ordinal" and len(sa[1]) == len(sb[1])
        level = "ordinal" if same_ordinal else "interval"
        fa = [np.nan if v is None else v for v in xa]
        fb = [np.nan if v is None else v for v in xb]
        both = [(u, v) for u, v in zip(xa, xb) if u is not None and v is not None]
        rel_a = [0 if v is None else 1 for v in xa]
        rel_b = [0 if v is None else 1 for v in xb]
        out.append({
            **p, "name_a": a["dims"][p["a"]]["fields"]["name"], "name_b": b["dims"][p["b"]]["fields"]["name"],
            "level": level, "mixed_value_spaces": sa[0] != sb[0] or (sa[0] == "ordinal" and not same_ordinal),
            "n_claims": len(claims), "n_both_relevant": len(both),
            "coverage_a": sum(rel_a) / len(claims) if claims else None,
            "coverage_b": sum(rel_b) / len(claims) if claims else None,
            "alpha_position": alpha(fa, fb, level,
                                    list(np.linspace(0, 1, len(sa[1]))) if same_ordinal else None),
            "spearman_position": spearman(fa, fb),
            "alpha_relevant_nominal": alpha(rel_a, rel_b, "nominal"),
            "position_sd_normalized": item_sd([u for u, _ in both], [v for _, v in both]),
        })
    return out


def score_relevance(a, b):
    items = sorted(set(a["grades"]) & set(b["grades"]))
    ga, gb = [a["grades"][i] for i in items], [b["grades"][i] for i in items]
    by_query = {}
    for i in items:
        q, rank, _ = i.split("|")
        by_query.setdefault(q, []).append((int(rank), a["grades"][i], b["grades"][i]))
    per_query, diffs = {}, []
    for q, rows in sorted(by_query.items()):
        rows.sort()
        na, nb = ndcg([r[1] for r in rows]), ndcg([r[2] for r in rows])
        k = cohen_kappa([r[1] for r in rows], [r[2] for r in rows])
        per_query[q] = {"n": len(rows), "kappa": k, "needs_adjudication": k is not None and k < 0.4 - 1e-9,
                        "ndcg_a": na, "ndcg_b": nb}
        if na is not None and nb is not None:
            diffs.append(na - nb)
    return {
        "n_items": len(items),
        "alpha_ordinal": alpha(ga, gb, "ordinal", [0, 1, 2, 3]),
        "kappa": cohen_kappa(ga, gb),
        "exact_agreement": sum(x == y for x, y in zip(ga, gb)) / len(items) if items else None,
        "grade_sd": item_sd(ga, gb),
        "per_query": per_query,
        "sd_of_per_query_ndcg_difference": float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else None,
    }


def score(path_a, path_b, match_path=None):
    a, b = read_sheet(path_a), read_sheet(path_b)
    pairs = read_match(match_path) if match_path else auto_match(a, b)
    dims = score_pair(a, b, pairs)
    matched_a = {p["a"] for p in pairs if p["same_axis"] == "yes"}
    matched_b = {p["b"] for p in pairs if p["same_axis"] == "yes"}
    n_a, n_b = len(a["dims"]), len(b["dims"])
    return {
        "sheet_a": str(path_a), "sheet_b": str(path_b),
        "match_source": str(match_path) if match_path else "identical name and definition",
        "errors_a": a["errors"], "errors_b": b["errors"],
        "minutes_a": a["minutes"], "minutes_b": b["minutes"],
        "contract_problems_a": {d: x["problems"] for d, x in a["dims"].items() if x["problems"]},
        "contract_problems_b": {d: x["problems"] for d, x in b["dims"].items() if x["problems"]},
        "axis_meaning": {
            "dims_a": n_a, "dims_b": n_b,
            "pairs_yes": sum(p["same_axis"] == "yes" for p in pairs),
            "pairs_partial": sum(p["same_axis"] == "partial" for p in pairs),
            "dice_matched": (len(matched_a) + len(matched_b)) / (n_a + n_b) if n_a + n_b else None,
        },
        "dimension_pairs": dims,
        "relevance": score_relevance(a, b),
    }


COMBINED_COLUMNS = [
    "record_type", "annotator_id", "blind_to_peer", "blind_to_key",
    "item_id", "query_id", "rank", "claim_id", "prompt", "dimension_id",
    "field_name", "value", "grade", "note", "match_dimension_a",
    "match_dimension_b", "axis_match", "direction", "match_filled_at", "match_note",
]


def wide_rows(rows, annotator_id):
    """Reconstruct the original sheet shape for one pseudonymous annotator."""
    fields, field_order = {}, []
    positions, position_order = {}, []
    relevance, times = [], []
    for row in rows:
        if row["annotator_id"] != annotator_id:
            continue
        kind = row["record_type"]
        if kind == "dimension_field":
            field = row["field_name"]
            if field not in fields:
                fields[field] = {"row_type": "DIM_FIELD", "item_id": field, "prompt": row["prompt"]}
                field_order.append(field)
            fields[field][row["dimension_id"]] = row["value"]
            fields[field]["note"] = row["note"]
        elif kind == "position":
            item = row["item_id"]
            if item not in positions:
                positions[item] = {"row_type": "POS", "item_id": item, "prompt": row["prompt"]}
                position_order.append(item)
            positions[item][row["dimension_id"]] = row["value"]
            positions[item]["note"] = row["note"]
        elif kind == "relevance":
            relevance.append({
                "row_type": "REL", "item_id": row["item_id"], "prompt": row["prompt"],
                "grade": row["grade"], "note": row["note"],
            })
        elif kind == "time":
            times.append({
                "row_type": "TIME", "item_id": row["item_id"], "prompt": "",
                "note": row["value"],
            })
    return [fields[x] for x in field_order] + [positions[x] for x in position_order] + relevance + times


def score_combined(path, side_a="A", side_b="B"):
    """Reconstruct two sides from one anonymous long-form CSV, then score them."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(COMBINED_COLUMNS) - set(rows[0]):
        missing = sorted(set(COMBINED_COLUMNS) - set(rows[0] if rows else []))
        raise ValueError(f"{path}: missing merged columns {missing}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a_path, b_path, match_path = root / "a.csv", root / "b.csv", root / "match.csv"
        rows_a = wide_rows(rows, side_a)
        rows_b = wide_rows(rows, side_b)
        if not rows_a or not rows_b:
            raise ValueError(f"{path}: comparison {side_a}:{side_b} has an empty side")
        with open(a_path, "w", newline="", encoding="utf-8") as a_handle, \
             open(b_path, "w", newline="", encoding="utf-8") as b_handle:
            writer_a = csv.DictWriter(a_handle, fieldnames=COLUMNS)
            writer_b = csv.DictWriter(b_handle, fieldnames=COLUMNS)
            writer_a.writeheader()
            writer_b.writeheader()
            writer_a.writerows(rows_a)
            writer_b.writerows(rows_b)

        matches = [
            row for row in rows
            if row["record_type"] == "match" and side_a == "A" and side_b == "B"
        ]
        match_arg = None
        if matches:
            with open(match_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "row_type", "dim_a", "dim_b", "same_axis", "direction",
                    "filled_by", "filled_at", "note",
                ])
                for row in matches:
                    writer.writerow([
                        "MATCH", row["match_dimension_a"], row["match_dimension_b"],
                        row["axis_match"], row["direction"], "coordinator",
                        row["match_filled_at"], row["match_note"],
                    ])
            match_arg = match_path

        report = score(a_path, b_path, match_arg)
    report["sheet_a"] = f"{path}:annotator_{side_a}"
    report["sheet_b"] = f"{path}:annotator_{side_b}"
    report["match_source"] = f"{path}:MATCH rows" if matches else "identical name and definition"
    return report


RESULT_COLUMNS = [
    "dataset", "group", "metric", "dimension_a", "dimension_b",
    "axis_match", "direction", "value", "n", "detail",
]


def result_rows(report, dataset):
    def row(group, metric, value, n="", detail="", pair=None):
        pair = pair or {}
        return {
            "dataset": dataset,
            "group": group,
            "metric": metric,
            "dimension_a": pair.get("a", ""),
            "dimension_b": pair.get("b", ""),
            "axis_match": pair.get("same_axis", ""),
            "direction": "reversed" if pair.get("reversed") else "",
            "value": "" if value is None else value,
            "n": n,
            "detail": detail,
        }

    out = []
    meaning = report["axis_meaning"]
    for metric in ("dims_a", "dims_b", "pairs_yes", "pairs_partial", "dice_matched"):
        out.append(row("axis_meaning", metric, meaning[metric]))
    for pair in report["dimension_pairs"]:
        if "error" in pair:
            out.append(row("dimension_pair", "error", "", detail=pair["error"], pair=pair))
            continue
        out.extend([
            row("dimension_pair", "alpha_position", pair["alpha_position"], pair["n_both_relevant"], pair=pair),
            row("dimension_pair", "spearman_position", pair["spearman_position"], pair["n_both_relevant"], pair=pair),
            row("dimension_pair", "coverage_a", pair["coverage_a"], pair["n_claims"], pair=pair),
            row("dimension_pair", "coverage_b", pair["coverage_b"], pair["n_claims"], pair=pair),
            row("dimension_pair", "alpha_relevant_nominal", pair["alpha_relevant_nominal"], pair["n_claims"], pair=pair),
            row("dimension_pair", "position_sd_mean", pair["position_sd_normalized"]["mean_item_sd"],
                pair["position_sd_normalized"]["n"], pair=pair),
            row("dimension_pair", "position_sd_pooled", pair["position_sd_normalized"]["pooled_item_sd"],
                pair["position_sd_normalized"]["n"], pair=pair),
        ])
    relevance = report["relevance"]
    out.extend([
        row("relevance", "alpha_ordinal", relevance["alpha_ordinal"], relevance["n_items"]),
        row("relevance", "kappa", relevance["kappa"], relevance["n_items"]),
        row("relevance", "exact_agreement", relevance["exact_agreement"], relevance["n_items"]),
        row("relevance", "grade_sd_mean", relevance["grade_sd"]["mean_item_sd"], relevance["grade_sd"]["n"]),
        row("relevance", "grade_sd_pooled", relevance["grade_sd"]["pooled_item_sd"], relevance["grade_sd"]["n"]),
        row("relevance", "sd_per_query_ndcg_difference",
            relevance["sd_of_per_query_ndcg_difference"], len(relevance["per_query"])),
    ])
    for query_id, values in relevance["per_query"].items():
        for metric in ("kappa", "ndcg_a", "ndcg_b", "needs_adjudication"):
            out.append(row("query", metric, values[metric], values["n"], detail=query_id))
    return out


def write_result_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v):
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_report(r):
    print(f"A: {r['sheet_a']}\nB: {r['sheet_b']}\nDimension pairing: {r['match_source']}")
    for side in ("a", "b"):
        for e in r[f"errors_{side}"]:
            print(f"  sheet {side.upper()} error: {e}")
        for d, probs in r[f"contract_problems_{side}"].items():
            print(f"  sheet {side.upper()} {d} contract: {'; '.join(probs)}")
        mins = r[f"minutes_{side}"]
        if mins:
            steps = ", ".join(f"{k} {v}" for k, v in mins.items())
            print(f"  sheet {side.upper()} time: {steps}; total {sum(mins.values())} min")
    m = r["axis_meaning"]
    print(f"\nAxis meaning: A authored {m['dims_a']}, B authored {m['dims_b']}; "
          f"matched pairs yes={m['pairs_yes']} partial={m['pairs_partial']}; "
          f"Dice share of Dimensions matched = {fmt(m['dice_matched'])}")
    for p in r["dimension_pairs"]:
        if "error" in p:
            print(f"  {p['a']}~{p['b']}: {p['error']}")
            continue
        sd = p["position_sd_normalized"]
        print(f"  {p['a']} '{p['name_a']}' ~ {p['b']} '{p['name_b']}' [{p['same_axis']}"
              f"{', reversed' if p['reversed'] else ''}{', mixed spaces' if p['mixed_value_spaces'] else ''}]")
        print(f"    alpha({p['level']}) = {fmt(p['alpha_position'])}, Spearman = {fmt(p['spearman_position'])}, "
              f"n both relevant = {p['n_both_relevant']}/{p['n_claims']}")
        print(f"    coverage A = {fmt(p['coverage_a'])}, B = {fmt(p['coverage_b'])}, "
              f"alpha(relevant, nominal) = {fmt(p['alpha_relevant_nominal'])}")
        print(f"    per-item position SD (0-1 scale): mean = {fmt(sd['mean_item_sd'])}, "
              f"pooled = {fmt(sd['pooled_item_sd'])}")
    rel = r["relevance"]
    print(f"\nRelevance ({rel['n_items']} items): alpha(ordinal) = {fmt(rel['alpha_ordinal'])}, "
          f"kappa = {fmt(rel['kappa'])}, exact agreement = {fmt(rel['exact_agreement'])}")
    print(f"  per-item grade SD (0-3 scale): mean = {fmt(rel['grade_sd']['mean_item_sd'])}, "
          f"pooled = {fmt(rel['grade_sd']['pooled_item_sd'])}")
    print(f"  SD of per-query nDCG difference between A and B (judgment noise only) = "
          f"{fmt(rel['sd_of_per_query_ndcg_difference'])}")
    for q, s in rel["per_query"].items():
        flag = "  <- kappa < 0.4, adjudicate" if s["needs_adjudication"] else ""
        print(f"  {q}: kappa = {fmt(s['kappa'])}, nDCG A = {fmt(s['ndcg_a'])}, B = {fmt(s['ndcg_b'])}{flag}")


# ---------- self-test ----------

def write_sheet(path, dims, positions, grades):
    """dims: {D: {field: value}}; positions: {claim: {D: raw}}; grades: {item: int}."""
    fields = ["name", "definition", "projection", "value_space", "comparison",
              "normalizer", "prototypes", "provenance", "version"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow(["HELP"] + ["synthetic"] * (len(COLUMNS) - 1))
        for fld in fields:
            w.writerow(["DIM_FIELD", fld, ""] + [dims.get(d, {}).get(fld, "") for d in DIMS] + ["", ""])
        for c, vals in positions.items():
            w.writerow(["POS", c, "claim text"] + [vals.get(d, "") for d in DIMS] + ["", ""])
        for item, g in grades.items():
            w.writerow(["REL", item, "query || claim", "", "", "", "", g, ""])
        w.writerow(["TIME", "author_minutes", "minutes", "", "", "", "", "", "17"])


def write_match(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "dim_a", "dim_b", "same_axis", "direction", "filled_by", "filled_at", "note"])
        for a, b, same, direction in rows:
            w.writerow(["MATCH", a, b, same, direction, "selftest", "", ""])


def selftest():
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' (' + detail + ')' if detail else ''}")

    print("1. krippendorff package against the published example (Krippendorff 2011, 4 coders x 12 units)")
    nan = np.nan
    ref = np.array([
        [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
        [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
    ])
    for level, expected in (("nominal", 0.743), ("interval", 0.849)):
        got = krippendorff.alpha(reliability_data=ref, level_of_measurement=level)
        check(f"{level} alpha = {expected}", abs(got - expected) < 0.001, f"got {got:.4f}")

    ordinal4 = {"name": "Feasibility", "definition": "How ready the step is",
                "projection": "annotator", "value_space": "ordinal: blocked < constrained < feasible < ready",
                "comparison": "rank distance", "normalizer": "divide by levels minus one",
                "prototypes": "permit refused = blocked; all in place = ready",
                "provenance": "selftest", "version": "1"}
    levels = ["blocked", "constrained", "feasible", "ready"]
    rng = random.Random(7)
    claims = [f"C{i:02d}" for i in range(1, 31)]
    truth = {c: rng.randrange(4) for c in claims}
    nr = set(claims[:4])
    pos = {c: {"D1": NR if c in nr else levels[truth[c]]} for c in claims}
    grades = {f"Q{q}|{k}|C{q}{k}": rng.randrange(4) for q in range(1, 5) for k in range(1, 7)}

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        print("2. identical sheets")
        write_sheet(t / "a.csv", {"D1": ordinal4}, pos, grades)
        write_sheet(t / "b.csv", {"D1": ordinal4}, pos, grades)
        r = score(t / "a.csv", t / "b.csv")
        p = r["dimension_pairs"][0]
        check("auto-match pairs the shared Dimension", r["axis_meaning"]["pairs_yes"] == 1)
        check("position alpha = 1.0 exactly", p["alpha_position"] == 1.0, fmt(p["alpha_position"]))
        check("ordinal level used for equal ordinal scales", p["level"] == "ordinal")
        check("coverage counts NR", abs(p["coverage_a"] - 26 / 30) < 1e-12, fmt(p["coverage_a"]))
        check("relevance alpha = 1.0 and kappa = 1.0",
              r["relevance"]["alpha_ordinal"] == 1.0 and r["relevance"]["kappa"] == 1.0)
        check("per-item SDs are zero", r["relevance"]["grade_sd"]["pooled_item_sd"] == 0.0
              and p["position_sd_normalized"]["pooled_item_sd"] == 0.0)
        check("no parse errors", not r["errors_a"] and not r["errors_b"] and not r["contract_problems_a"])
        check("TIME rows read", r["minutes_a"] == {"author_minutes": 17})

        print("3. hand-computed per-item SD")
        ga = {"Q1|1|C1": 3, "Q1|2|C2": 2, "Q1|3|C3": 1, "Q1|4|C4": 0}
        gb = {"Q1|1|C1": 1, "Q1|2|C2": 2, "Q1|3|C3": 1, "Q1|4|C4": 3}
        write_sheet(t / "a.csv", {"D1": ordinal4}, pos, ga)
        write_sheet(t / "b.csv", {"D1": ordinal4}, pos, gb)
        sd = score(t / "a.csv", t / "b.csv")["relevance"]["grade_sd"]
        # per-item SDs: |3-1|/sqrt2, 0, 0, |0-3|/sqrt2 -> mean 5/(4 sqrt2), pooled sqrt((2+4.5)/4)
        check("mean per-item SD = 0.8839", abs(sd["mean_item_sd"] - 5 / (4 * math.sqrt(2))) < 1e-12,
              fmt(sd["mean_item_sd"]))
        check("pooled per-item SD = 1.2748", abs(sd["pooled_item_sd"] - math.sqrt(6.5 / 4)) < 1e-12,
              fmt(sd["pooled_item_sd"]))

        print("4. reversed scalar Dimension matched to an ordinal one through the match sheet")
        scalar_rev = {**ordinal4, "name": "Distance from launch", "definition": "Inverse readiness",
                      "value_space": "scalar: 0-1", "comparison": "absolute difference",
                      "normalizer": "divide by range", "prototypes": "ready to go = 0; refused = 1"}
        pos_b = {c: {"D2": NR if c in nr else f"{1 - truth[c] / 3:.6f}"} for c in claims}
        write_sheet(t / "a.csv", {"D1": ordinal4}, pos, grades)
        write_sheet(t / "b.csv", {"D2": scalar_rev}, pos_b, grades)
        write_match(t / "m.csv", [("D1", "D2", "yes", "reversed")])
        p = score(t / "a.csv", t / "b.csv", t / "m.csv")["dimension_pairs"][0]
        check("interval alpha after flip = 1.0", abs(p["alpha_position"] - 1.0) < 1e-9, fmt(p["alpha_position"]))
        check("pair flagged as mixed value spaces", p["mixed_value_spaces"] and p["level"] == "interval")

        print("5. partial and no agreement")
        noisy = {c: {"D1": NR if c in nr else levels[min(3, max(0, truth[c] + rng.choice([-1, 0, 0, 1])))]}
                 for c in claims}
        noisy_grades = {k: min(3, max(0, g + rng.choice([-1, 0, 0, 1]))) for k, g in grades.items()}
        write_sheet(t / "a.csv", {"D1": ordinal4}, pos, grades)
        write_sheet(t / "b.csv", {"D1": ordinal4}, noisy, noisy_grades)
        r = score(t / "a.csv", t / "b.csv")
        a_noisy = r["dimension_pairs"][0]["alpha_position"]
        check("noisy copy: 0.3 < position alpha < 1", 0.3 < a_noisy < 1.0, fmt(a_noisy))
        check("noisy copy: 0.3 < relevance alpha < 1", 0.3 < r["relevance"]["alpha_ordinal"] < 1.0,
              fmt(r["relevance"]["alpha_ordinal"]))
        check("nDCG difference SD is reported", r["relevance"]["sd_of_per_query_ndcg_difference"] is not None)
        indep = {c: {"D1": NR if c in nr else levels[rng.randrange(4)]} for c in claims}
        write_sheet(t / "b.csv", {"D1": ordinal4}, indep, grades)
        a_ind = score(t / "a.csv", t / "b.csv")["dimension_pairs"][0]["alpha_position"]
        check("independent positions: alpha below the noisy copy and below 0.4", a_ind < min(a_noisy, 0.4),
              fmt(a_ind))

        print("6. contract and value checks")
        bad = {**ordinal4, "value_space": "nominal: red, green, blue"}
        write_sheet(t / "b.csv", {"D1": bad}, pos, grades)
        errs = read_sheet(t / "b.csv")["errors"]
        check("unordered categories rejected", any("unordered" in e for e in errs))
        wrong = {**ordinal4, "comparison": "absolute difference", "prototypes": "only one = ready"}
        bad_pos = {**pos, "C10": {"D1": "maybe"}, "C11": {"D1": ""}}
        write_sheet(t / "b.csv", {"D1": wrong}, bad_pos, {**grades, "Q1|1|C11": "x"})
        s = read_sheet(t / "b.csv")
        probs = s["dims"]["D1"]["problems"]
        check("comparison/value-space mismatch flagged", any("rank distance" in x for x in probs))
        check("prototype count flagged", any("prototypes" in x for x in probs))
        check("invalid level, blank position and bad grade reported", len(s["errors"]) == 3, str(len(s["errors"])))

    ok = all(checks)
    print(f"\nselftest: {sum(checks)}/{len(checks)} checks passed -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
    args = [a for a in args if a not in opts.values()]
    if len(args) != 1:
        print(__doc__)
        return 2
    compare = opts.get("--compare", "A:B").split(":")
    if len(compare) != 2:
        print("--compare must be SIDE_A:SIDE_B", file=sys.stderr)
        return 2
    report = score_combined(Path(args[0]), compare[0], compare[1])
    print_report(report)
    if "--csv" in opts:
        write_result_csv(opts["--csv"], result_rows(report, opts.get("--dataset", Path(args[0]).stem)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
