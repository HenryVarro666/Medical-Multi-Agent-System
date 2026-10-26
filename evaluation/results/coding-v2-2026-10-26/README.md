# Coding-validation v2 · October 26, 2026

This archive retains a new real Codex-CLI comparison after patient-context, official code-validation and reference-granularity repairs. Models are gpt-6.1-sol and gpt-6-astra, each evaluated once on unchanged case_001/case_002 narratives. The selected complete batches both returned five Agent sections without pipeline/harness errors.

## Shared protocol

- Clean source: 426d108f0c5f2c5af398b939970186526d717878; Python 3.12.13; Codex CLI 0.163.0.
- Dataset: data/evaluation_cases_v2.json, version coding-v2-fy2027, scorer coding-validation-v2.
- Same medium reasoning, 600-second per-call timeout, zero diagnosis loopbacks, serial batches and cases. No model-output or validation repair retries are hidden within an invocation.
- Packaged official CMS FY2027 catalog and source/member/payload hashes are recorded in each summary and comparison.json. Runtime catalog validation is offline and tests all primary/secondary proposals for canonical syntax, membership and complete-code status. It does not establish clinical correctness.

The original system text becomes the CLI instructions file; human text is supplied on stdin. User configuration is ignored, tools disabled, and any observed tool activity causes rejection. Temperature and resolved backend revision are unavailable. Both model names are requested identifiers. Raw CLI startup/reasoning/authentication logs are not published.

## Reference revision and comparison limits

The original five-case file and earlier result artifacts remain unchanged. V2 pins FY2027 and retains `J18.1` as a provisional full-code pneumonia target. It refines the non-reportable I21.0 parent to I21.09 using the official Index anterior-STEMI entry and Tabular anterior-wall NOS inclusion, without assuming a culprit artery. This engineering review followed the v1 score review; it is not independent clinical coding adjudication. The diagnosis-level ambiguity for pneumonia and unspecified encounter setting remain limitations. No reference labels or revision notes enter model prompts.

`label_transition.json` separately reports original v1 exact scores, post-hoc v1-output rescoring against the v2 references, new v2 exact scores, and frozen-legacy-label scores. Both old and new predictions score 1/2 against v2 references, while the original archived v1 exact score remains 0/2. The percentage change is not demonstrated primary-code improvement.

Coding now receives the exact narrative, full structured patient record, Diagnosis, Treatment, and an explicit research coding contract. Saved prompts prove context preservation; the output contains original model proposals/raw text, canonical code descriptions, and deterministic validation reports. Clinical review remains required even when every code is present and complete.

## Reports and retained attempts

- [gpt-6.1-sol](gpt-6.1-sol/eval_results.md): 2/2 complete pipelines; v2 exact 1/2; all-code catalog validation 2/2; mean 101.79 seconds.
- [gpt-6-astra](gpt-6-astra/eval_results.md): 2/2 complete pipelines; v2 exact 1/2; all-code catalog validation 2/2; mean 126.77 seconds.
- [Interrupted first Sol batch](attempts/README.md): two attempted pipelines failed in native Codex transport before Coding, overlapping host sleep/wake. The full batch was repeated after recovery with unchanged source/settings; first-attempt observations and raw text remain archived.

Across all three batches, four of six pipeline attempts completed without errors. The charts compare the two complete-stage batches; they are not an all-attempt availability or unconditional latency estimate. `comparison.json` records chronological attempts and selections. Sleep-affected timestamps/active-clock durations in the interrupted cohort must not be interpreted as normal model timing.

[SHA256SUMS](SHA256SUMS) covers the stored evidence. Raw files include all stage outputs and captured model-call messages, hashes, usage and completion events when available. Failed/missing responses remain failures in their own reports.

## Reproduce into new directories

Authenticate Codex with your own account, install the committed dependency lock, and run from the repository root:

```bash
for model in gpt-6.1-sol gpt-6-astra; do
  LLM_BACKEND=codex CODEX_MODEL="$model" CODEX_REASONING_EFFORT=medium \
  MAX_DIAGNOSIS_RETRIES=0 LLM_TIMEOUT_SECONDS=600 \
  uv run --locked python -m evaluation.run_eval \
    --dataset data/evaluation_cases_v2.json --cases case_001,case_002 --repeats 1
done
```

Regenerate plots from the saved summaries without model calls:

```bash
uv run --locked --extra plots python -m evaluation.plot_comparison \
  --results evaluation/results/coding-v2-2026-10-26/gpt-6.1-sol/eval_results.json \
  evaluation/results/coding-v2-2026-10-26/gpt-6-astra/eval_results.json \
  --prefix coding-v2-2026-10-26
```

Model availability, backend conditions and outputs may change. Two synthetic cases and one repeat do not support a clinical performance or general model ranking claim.
