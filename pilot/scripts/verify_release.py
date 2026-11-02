#!/usr/bin/env python3
"""Check the public result tables against the paper's reported snapshots."""

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_SHA = "413b34d57539da0a8971a86c18a2abf533d9493d4dddf997c60e78e5358278ab"

TABLE8 = {
    "i": ("0.421", "0.199", "0.609", "0.400"),
    "ii": ("0.453", "0.193", "0.627", "0.417"),
    "iii": ("0.481", "0.199", "0.656", "0.441"),
    "iv": ("0.456", "0.213", "0.637", "0.422"),
    "v": ("0.400", "0.222", "0.582", "0.378"),
    "vi": ("0.372", "0.207", "0.562", "0.349"),
    "vii": ("0.450", "0.219", "0.643", "0.428"),
    "viii": ("0.428", "0.194", "0.605", "0.397"),
    "ix": ("0.464", "0.224", "0.639", "0.426"),
}

GAINS = {
    "G01": ("272", "0.028", "0.014", "0.056", "0.002"),
    "G02": ("272", "0.032", "0.004", "0.052", "0.011"),
    "G03": ("272", "0.021", "0.001", "0.043", "0.020"),
    "G04": ("251", "0.081", "0.058", "0.112", "<0.001"),
    "G05": ("251", "0.053", "0.033", "0.083", "<0.001"),
    "G06": ("272", "0.049", "0.024", "0.076", "<0.001"),
    "G07": ("272", "0.031", "0.009", "0.055", "0.004"),
    "G08": ("91; 88; 93", "not_reported", "-0.018", "not_reported", "not_applicable (bound test)"),
    "G09": ("272", "0.11", "not_applicable (point comparison)", "not_applicable (point comparison)", "not_applicable (point comparison)"),
    "G10": ("272", "0.011", "-0.006", "0.036", "not_applicable (interval only)"),
    "G11": ("272", "0.017", "0.000", "0.040", "not_applicable (interval only)"),
    "G12": ("272", "0.39", "not_applicable (point comparison)", "not_applicable (point comparison)", "not_applicable (point comparison)"),
    "G13": ("58", "0.061", "0.018", "0.104", "0.003"),
}


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main():
    errors = 0
    plan = (ROOT / "evaluation-plan.md").read_bytes()
    digest = hashlib.sha256(plan).hexdigest()
    if digest != PLAN_SHA:
        errors += fail(f"plan sha256 {digest}")

    record_text = (ROOT / "execution_record.yaml").read_text()
    component_ids = []
    for line in record_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- {id: C") or stripped.startswith("- id: C") or stripped.startswith("id: C"):
            token = stripped.split("id: ", 1)[1][:3]
            if token.startswith("C"):
                component_ids.append(token)
    if component_ids != [f"C{number:02d}" for number in range(1, 20)]:
        errors += fail(f"component ids {component_ids}")
    if record_text.count("status: scored_partial_record") != 5:
        errors += fail("expected four components plus arm ix marked scored_partial_record")
    if "id: arm_ix" not in record_text or "registration_relation: post_hoc_unregistered" not in record_text:
        errors += fail("arm ix registration")
    if "scored_at:" in record_text:
        errors += fail("uncorroborated scored_at")

    results = list(csv.DictReader((ROOT / "results/retrieval_results.csv").open()))
    if len(results) != 27:
        errors += fail(f"expected 27 pooled rows, got {len(results)}")
    by_arm = {}
    for row in results:
        by_arm.setdefault(row["arm_id"], {})[row["metric"]] = row
        if row["metric"] == "nDCG@10" and row["metric_role"] != "primary":
            errors += fail(f"{row['arm_id']} nDCG role")
        if row["metric"] == "Recall@50" and (row["metric_role"] != "secondary" or row["sd_per_query"]):
            errors += fail(f"{row['arm_id']} Recall@50")
        if row["metric"] == "MRR" and (row["metric_role"] != "exploratory" or row["sd_per_query"]):
            errors += fail(f"{row['arm_id']} MRR")
    for arm, (ndcg, sd, recall, mrr) in TABLE8.items():
        got = by_arm[arm]
        if (got["nDCG@10"]["mean"], got["nDCG@10"]["sd_per_query"], got["Recall@50"]["mean"], got["MRR"]["mean"]) != (ndcg, sd, recall, mrr):
            errors += fail(f"table 8 mismatch for arm {arm}")
        expected_status = "post_hoc_unregistered" if arm == "ix" else "registered"
        if any(got[metric]["registration_status"] != expected_status for metric in got):
            errors += fail(f"arm {arm} status")

    gains = list(csv.DictReader((ROOT / "results/retrieval_gains.csv").open()))
    ids = [row["gain_id"] for row in gains]
    if ids != [f"G{number:02d}" for number in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]:
        errors += fail(f"gain ids {ids}")
    for row in gains:
        expected = GAINS[row["gain_id"]]
        got = (row["n_queries"], row["mean_difference"], row["ci_lower"], row["ci_upper"], row["p_value"])
        if got != expected:
            errors += fail(f"{row['gain_id']} values {got}")
        if "0.021 to 0.039" in " ".join(row.values()):
            errors += fail("unsupported G08 range")
        if "amendment_pending" in " ".join(row.values()):
            errors += fail("amendment_pending remains")
    outcomes = {row["gain_id"]: row["reported_outcome"] for row in gains}
    if outcomes["G01"] != "passes_primary_bar_on_recovered_values":
        errors += fail("G01 outcome")
    if outcomes["G08"] != "passes_on_recovered_worst_lower_bound":
        errors += fail("G08 outcome")
    if outcomes["G12"] != "passes_proposed_post_hoc_bar":
        errors += fail("G12 outcome")

    arms = list(csv.DictReader((ROOT / "results/retrieval_ablation_arms.csv").open()))
    if [row["arm_id"] for row in arms] != list("i ii iii iv v vi vii viii ix".split()):
        errors += fail("arm inventory")
    for row in arms:
        if row["arm_id"] == "ix":
            if row["registered"] != "post_hoc_unregistered" or row["role_in_plan"] != "post_hoc_attribution_check":
                errors += fail("arm ix label")
        elif row["registered"] != "yes":
            errors += fail(f"arm {row['arm_id']} should be registered")
        if "delta_d(pi_d(c) pi_d(q))" in row["dimension_features"]:
            errors += fail("missing comma in delta_d")

    validity = list(csv.DictReader((ROOT / "results/dimension_validity.csv").open()))
    if [row["pair_id"] for row in validity] != ["P1", "P2", "P3", "P4", "P5", "R1"]:
        errors += fail("validity rows")
    for row in validity:
        if row["registered_status"] != "descriptive_pilot_unregistered":
            errors += fail(f"{row['pair_id']} status")
        if row["source"] != "pilot/pilot_results.csv":
            errors += fail(f"{row['pair_id']} source")

    banned = ["evidence-ledger.csv", "evidence_ledger.csv", "validity-index.csv", "validity_index.csv"]
    for name in banned:
        if (ROOT / "results" / name).exists():
            errors += fail(f"banned file {name}")

    if errors:
        print(f"{errors} release checks failed", file=sys.stderr)
        return 1
    print("PASS: release tables match the reported paper snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
