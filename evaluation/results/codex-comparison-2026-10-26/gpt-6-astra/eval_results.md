# Synthetic-vignette evaluation

- Run at: 2026-10-26T06:47:53.647609+00:00
- Model: `gpt-6-astra`; backend: `Codex CLI (saved ChatGPT authentication)`
- Cases: 2; repeats: 1; runs: 2
- Source revision: `347f414120f30a837ca927617eeb8f89102ccee2`; uncommitted changes: False
- Raw pipeline results: `raw/`

| Case | Repeat | Diagnosis alias tier | ICD exact | ICD category | Latency (s) |
|---|---:|---|---|---|---:|
| case_001 | 1 | strict | False | True | 197.84 |
| case_002 | 1 | strict | False | True | 215.04 |

| Metric | Hits / runs | Fraction |
|---|---:|---:|
| diagnosis_top1_strict | 2 / 2 | 1.0 |
| diagnosis_top1_any | 2 / 2 | 1.0 |
| diagnosis_top3_any | 2 / 2 | 1.0 |
| icd10_exact | 0 / 2 | 0.0 |
| icd10_category | 2 / 2 | 1.0 |
| treatment_structure_complete | 2 / 2 | 1.0 |
| ddi_engine_checked | 2 / 2 | 1.0 |
| audit_completed | 2 / 2 | 1.0 |

Mean pipeline latency: **206.44 s**

## Interpretation

These are demonstration measurements on synthetic fixtures, not clinical validation.
Diagnosis tiers use case-specific substring aliases without negation handling.
Repeated runs reuse the same cases; they do not increase clinical case diversity.
The small label set has not been independently reviewed by clinicians or coders.
`treatment_structure_complete` checks for medications and a DDI execution stamp;
it does not assess drug selection, dosage, completeness, or safety.
`audit_completed` means a report exists; it does not establish HIPAA compliance.
Latency depends on the backend and hardware; no throughput claim is made.
