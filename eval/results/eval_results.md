# Eval results — medical multi-agent pipeline

- run at: 2026-07-13T00:45:58.494029+00:00
- backend: `http://localhost:8000/v1`, model: `Qwen/Qwen3-14B`
- cases: 5 labeled vignettes x 2 repeat(s) = 10 pipeline runs
- every value below comes from an actual pipeline run (raw outputs in `eval/results/raw/`)

## Per-case results

| case | expected dx | predicted dx | top-1 | top-3 | expected ICD | predicted ICD | ICD exact | ICD cat | treat OK | DDI ran | audit | lat (s) |
|------|-------------|--------------|-------|-------|--------------|---------------|-----------|---------|----------|---------|-------|---------|
| case_001 r1 | Community-Acquired Pneumonia | Community-acquired pneumonia | strict | yes | J18.1 | J18.9 | no | yes | yes | yes | yes | 27.1 |
| case_002 r1 | ST-Elevation Myocardial Infarction (STEMI) | Acute myocardial infarction | compatible | yes | I21.0 | I21.9 | no | yes | yes | yes | yes | 29.16 |
| case_003 r1 | Hypothyroidism (Hashimoto's Thyroiditis) | Hypothyroidism | strict | yes | E03.9 | E03.9 | yes | yes | yes | yes | yes | 24.38 |
| case_004 r1 | Acute Decompensated Heart Failure | Heart Failure with Reduced Ejection Fraction (HFrEF) | compatible | yes | I50.9 | I50.21 | no | yes | yes | yes | yes | 35.92 |
| case_005 r1 | Acute Appendicitis | Appendicitis | strict | yes | K35.80 | K35.80 | yes | yes | yes | yes | yes | 24.29 |
| case_001 r2 | Community-Acquired Pneumonia | Community-acquired pneumonia | strict | yes | J18.1 | J18.9 | no | yes | yes | yes | yes | 27.48 |
| case_002 r2 | ST-Elevation Myocardial Infarction (STEMI) | Acute myocardial infarction | compatible | yes | I21.0 | I21.9 | no | yes | yes | yes | yes | 30.52 |
| case_003 r2 | Hypothyroidism (Hashimoto's Thyroiditis) | Hypothyroidism | strict | yes | E03.9 | E03.9 | yes | yes | yes | yes | yes | 26.83 |
| case_004 r2 | Acute Decompensated Heart Failure | Congestive Heart Failure (CHF) | compatible | yes | I50.9 | I50.9 | yes | yes | yes | yes | yes | 33.62 |
| case_005 r2 | Acute Appendicitis | Appendicitis | strict | yes | K35.80 | K35.80 | yes | yes | yes | yes | yes | 22.9 |

## Aggregate

| metric | hits / n | fraction |
|--------|----------|----------|
| diagnosis_top1_strict | 6 / 10 | 0.6 |
| diagnosis_top1_any | 10 / 10 | 1.0 |
| diagnosis_top3_any | 10 / 10 | 1.0 |
| icd10_exact | 5 / 10 | 0.5 |
| icd10_category | 10 / 10 | 1.0 |
| treatment_ok | 10 / 10 | 1.0 |
| ddi_engine_checked | 10 / 10 | 1.0 |
| audit_completed | 10 / 10 | 1.0 |

Mean pipeline latency: **28.22 s**

## Honest caveats

- N = 5 labeled cases: demo-scale. These fractions are NOT
  population-level accuracy claims; with N=5, one case swings a metric by 20 points.
- Diagnosis matching uses the explicit alias table in `eval/run_eval.py`
  (`strict` = same clinical entity, `compatible` = clinically correct but coarser).
- ICD-10 labels are taken from `python/data/sample_patients.json` as-is;
  some labels are themselves debatable (e.g. case_003 is labeled E03.9 while
  a coder might defend E06.3 for Hashimoto's thyroiditis).
- `hipaa_compliant_flag` is usually false because the demo PHI name regex
  flags capitalized bigrams (a known precision limitation, documented in code).
