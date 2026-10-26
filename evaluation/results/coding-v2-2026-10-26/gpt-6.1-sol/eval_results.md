# Synthetic-vignette evaluation

- Run at: 2026-10-26T22:25:11.375936+00:00
- Model: `gpt-6.1-sol`; backend: `Codex CLI (saved ChatGPT authentication)`
- Cases: 2; repeats: 1; runs: 2
- Source revision: `426d108f0c5f2c5af398b939970186526d717878`; uncommitted changes: False
- Dataset: `coding-v2-fy2027`; scorer: `coding-validation-v2`
- Raw pipeline results: `raw/`

| Case | Repeat | Diagnosis alias tier | ICD exact | ICD category | Latency (s) |
|---|---:|---|---|---|---:|
| case_001 | 1 | compatible | False | True | 89.57 |
| case_002 | 1 | strict | True | True | 114.01 |

| Metric | Hits / runs | Fraction |
|---|---:|---:|
| diagnosis_top1_strict | 1 / 2 | 0.5 |
| diagnosis_top1_any | 2 / 2 | 1.0 |
| diagnosis_top3_any | 2 / 2 | 1.0 |
| icd10_exact | 1 / 2 | 0.5 |
| icd10_category | 2 / 2 | 1.0 |
| treatment_structure_complete | 2 / 2 | 1.0 |
| ddi_engine_checked | 2 / 2 | 1.0 |
| audit_completed | 2 / 2 | 1.0 |
| icd10_primary_catalog_valid | 2 / 2 | 1.0 |
| icd10_all_codes_catalog_valid | 2 / 2 | 1.0 |

Mean pipeline latency: **101.79 s**

## Interpretation

These are demonstration measurements on synthetic fixtures, not clinical validation.
Diagnosis tiers use case-specific substring aliases without negation handling.
Repeated runs reuse the same cases; they do not increase clinical case diversity.
The small label set has not been independently reviewed by clinicians or coders.
`treatment_structure_complete` checks for medications and a DDI execution stamp;
it does not assess drug selection, dosage, completeness, or safety.
`audit_completed` means a report exists; it does not establish HIPAA compliance.
Latency depends on the backend and hardware; no throughput claim is made.
Catalog validity verifies complete-code membership, not clinical correctness.
The revised exact-match reference is not directly comparable with the old label set.
