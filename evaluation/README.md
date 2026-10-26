# Evaluation

The runner executes the five synthetic cases in `data/sample_patients.json` through the configured five-stage pipeline. It records raw outputs, rule-execution stamps, errors, alias-based scores, ICD-10 matches, and wall-clock latency. It is an engineering demonstration, not a clinical benchmark.

## Run a new evaluation

From the repository root, install the package and configure `.env` as described in the main README. A working LLM backend is required for a live run.

```bash
python -m evaluation.run_eval --help
python -m evaluation.run_eval --repeats 2
python -m evaluation.run_eval --cases case_001,case_003 --repeats 1
python -m evaluation.run_eval --output-dir /tmp/clinical-eval-new
```

`--repeats` must be a positive integer. `--cases` accepts known, unique fixture IDs separated by commas. The default selects all five cases. `--help` and importing the scoring functions do not load LangGraph, construct an LLM client, or contact a backend.

By default, each run creates a new directory under `evaluation/results/runs/` with:

```text
run-<timestamp>-<suffix>/
├── eval_results.json
├── eval_results.md
└── raw/<case_id>_r<repeat>.json
```

A custom output directory must be new or empty. The runner rejects paths inside the historical archive and does not overwrite result files. New local results are ignored by Git. The process exits nonzero if a pipeline or harness error occurred; a diagnosis mismatch alone does not change its exit status.

Output directories are checked for writability before inference begins. If malformed model output prevents scoring, the raw stage results are still archived and the scoring exception is recorded in `harness_error`. That run exits nonzero.

Metadata includes timestamp, selected backend and model, Python version, fixture SHA-256, Git revision, and whether uncommitted changes were present. For a reproducible report, also record the model revision, serving command and versions, hardware, decoding configuration, and the exact source diff when dirty. Those details are not all captured automatically.

## Score definitions

| Field | Definition and limit |
|---|---|
| `diagnosis_top1_strict` | Primary name matches a case-specific substring in the historical strict alias tier |
| `diagnosis_top1_any` | Primary name matches either strict or compatible aliases |
| `diagnosis_top3_any` | Either alias tier matches the primary name or first two differentials |
| `icd10_exact` | Case-insensitive equality with the fixture label |
| `icd10_category` | Code prefix before the decimal matches; this is a category, not a chapter |
| `treatment_structure_complete` | Medications are present and the deterministic DDI check stamp is true |
| `ddi_engine_checked` | DDI execution stamp is true |
| `audit_completed` | A nonempty audit result exists |
| `latency_seconds` | Wall-clock duration of the entire pipeline invocation, including retries |

The historical alias rules remain explicit in `run_eval.py`. Substring matching has no negation or contextual reasoning and can accept overly broad names. The tier names do not certify diagnostic equivalence. The labels have not been independently reviewed by clinicians or coders. Repeated measurements reuse the same five cases and must not be counted as ten independent clinical examples.

Treatment structure checks do not evaluate drug selection, dosage, contraindication coverage, or overall safety. Audit completion does not prove de-identification or HIPAA compliance. There is no single-agent baseline, retrieval ablation, held-out evaluation, or demonstrated performance improvement from GraphRAG.

## Historical July 13, 2026 run

The original files are preserved byte-for-byte in [results/historical-2026-07-13](results/historical-2026-07-13/), including the summary and ten raw outputs. They record `Qwen/Qwen3-14B` at `http://localhost:8000/v1`, five cases, two repeats, strict alias matches of 6/10, either-tier matches of 10/10, exact ICD-10 matches of 5/10, and mean latency of 28.22 seconds.

The original historical field `treatment_ok` checked only output structure and DDI execution. New runs call the same structural measure `treatment_structure_complete` to make its limit explicit. Historical audit outputs also contain controls marked verified by the old code; the current audit node reports those deployment controls as unverified.

The historical metadata does not pin the model revision, dependency versions, serving configuration, or hardware. It is a preserved observation and cannot establish exact reproduction or current-code performance. Original Markdown links mention paths from the former repository layout; raw JSON and summaries were not rewritten during the refactor.

## Offline verification

### README figures

The README embeds aggregate metric and paired latency charts generated directly from the preserved JSON. PNG files provide the inline previews; SVG files provide vector versions. The figures retain the historical date, model, case count, repeat count, and interpretation limits.

Regenerate them without calling an LLM:

```bash
uv run --locked --extra plots python -m evaluation.plot_results
```

The optional `plots` extra supplies Matplotlib. Outputs go to `docs/assets/evaluation/`; `--results` and `--output-dir` select a different evaluation JSON and destination. Original result files are read only. The first five bars represent label matching; the final three represent output structure and stage execution.

### Case walkthrough figure

The `case_001` flowchart connects the archived input, five key stage outputs, and final scoring result. It reads `raw/case_001_r1.json` and the matching scoring row; it does not execute the pipeline or call a model.

```bash
uv run --locked --extra plots python -m evaluation.draw_case
```

Outputs are written as HTML, SVG, and PNG to `docs/assets/examples/`. Use `--output-dir` for a different destination. The single observed path includes one diagnosis attempt, the exact ICD-10 mismatch, and the historical audit limitations.

### Evaluation tests

```bash
python -m pytest tests/test_evaluation.py
```

These tests exercise scoring, argument validation, mocked successes and failures, output placement, and help without site packages. They do not make model calls or repeat the historical experiment.
