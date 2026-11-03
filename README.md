# Geometric Knowledge Evaluations

Registered plan, pilot judgments, and retrieval results for *Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex* (Miracle, Georgiou, Okoth, and Monday, 2026).

Nineteen components are registered. The Dimension term, standing penalty, Domain scoping, and HyDE escalation are scored on 272 pooled held-out queries in arms (i)–(viii). Arm (ix), facet-only, is scored in the same run. The pilot is annotators A and B on two public domains, plus a calibration set. Component status is in `execution_record.yaml`.

## Contents

| Path | What it holds |
| --- | --- |
| `evaluation-plan.md` | The evaluation specification. Tag `registered-plan` (2026-10-20) holds the registered text, SHA-256 `413b34d57539da0a8971a86c18a2abf533d9493d4dddf997c60e78e5358278ab`. |
| `AMENDMENTS.md` | Earlier versions of the plan. |
| `execution_record.yaml` | Status of the nineteen components, the reported contrasts, and arm (ix). |
| `pilot/` | A/B judgments (`calibration.csv`, `peps.csv`, `enron.csv`), protocol, provenance, normalized inputs, and `pilot_results.csv` (149 metric rows). |
| `pilot/scripts/` | Scoring and the checks below. |
| `results/` | Pooled arm means, gains, ablation arms, and the descriptive pilot validity table. |

The release contains the summary record, the pilot judgments, the normalized inputs, and the scripts.

## Use

Requirements: Python 3 with NumPy 2.5.4 and krippendorff 0.9.0.

```sh
python pilot/scripts/score.py --selftest
python pilot/scripts/score.py pilot/peps.csv --compare A:B --dataset peps --csv /tmp/peps.csv
python pilot/scripts/score.py pilot/enron.csv --compare A:B --dataset enron --csv /tmp/enron.csv
python pilot/scripts/score.py pilot/calibration.csv --compare A:B --dataset calibration_ab --csv /tmp/calibration_ab.csv
python pilot/scripts/score.py pilot/calibration.csv --compare B:KEY --dataset calibration_blind_key --csv /tmp/calibration_blind_key.csv
python pilot/scripts/verify_results.py --expected pilot/pilot_results.csv /tmp/calibration_ab.csv /tmp/calibration_blind_key.csv /tmp/peps.csv /tmp/enron.csv
python pilot/scripts/verify_release.py
```

`verify_results.py` checks the four scored files against `pilot/pilot_results.csv`. `verify_release.py` checks `results/` and `execution_record.yaml` against the figures reported in the paper. Retrieval rows are per arm and per contrast. There is no single summary score.

## Licences

This repository states no licence. The scorer imports krippendorff 0.9.0 (GPL-3.0-or-later). Selected PEP material is public domain or CC0-1.0-Universal; PEP 600 follows the PEP 1 policy. The CMU Enron release carries no written licence in the source archive. Cite the paper with `CITATION.cff`. Works cited by the plan are listed at the end of `evaluation-plan.md`.
