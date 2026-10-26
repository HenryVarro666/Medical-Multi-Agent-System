# Two-case Codex model comparison · October 26, 2026

Two original synthetic fixtures, case_001 and case_002, were each run once with gpt-6.1-sol and gpt-6-astra using the user’s ChatGPT-authenticated Codex CLI. These are real model outputs. All four pipelines returned five Agent sections with no pipeline or harness errors. Each used four model invocations and deterministic Audit.

## Protocol and provenance

Both models used medium reasoning, a 600-second per-call timeout, and zero diagnosis loopbacks. Fixed fixtures cannot add new observations; a request for more information is retained in the model trace but does not rerun Intake. Engineering completion therefore does not imply that clinical information is sufficient.

Model runs and cases were sequential, in model order gpt-6.1-sol then gpt-6-astra. They ran from a clean isolated clone of source `347f414120f30a837ca927617eeb8f89102ccee2` with the same fixture and dependency hashes, Python 3.12.13, and Codex CLI 0.163.0. `comparison.json` records the shared protocol. Per-model metadata records start/end times, settings, and source state.

The original system prompt is supplied as the CLI instructions file; the original human message is sent on stdin. Calls start in fresh temporary directories with user configuration ignored, tool features disabled, and observed tool activity rejected. The CLI/backend may add context. Temperature is not applied and the resolved backend model revision is unavailable. Model names are requested identifiers.

Raw files contain original narratives, all four model calls with messages, content, hashes, CLI usage/events when available, plus the five resulting Agent sections and deterministic enrichment. No authentication files or raw CLI startup/reasoning logs are published. The allowed skill-catalog omission notice is stored as a safe warning code.

## Reports

- [gpt-6.1-sol report](gpt-6.1-sol/eval_results.md) · mean 156.91 seconds.
- [gpt-6-astra report](gpt-6-astra/eval_results.md) · mean 206.44 seconds.
- [Interrupted ultra pilot](exploratory-astra-ultra/README.md), preserved separately and excluded from the matched comparison.
- [File checksums](SHA256SUMS).

Each model has strict/either-tier diagnosis matches of 2/2, exact ICD-10 matches of 0/2, category matches of 2/2, and treatment-structure/DDI/Audit completion checks of 2/2. Substring aliases and fixture labels are not clinical adjudication. Output structure does not measure treatment safety or compliance. Two cases and one repeat do not support uncertainty estimates or a general model ranking. Wall times include CLI/authentication/transport/inference overhead and may depend on backend conditions. Historical Qwen observations use different settings and case coverage.

## Reproduce

Use the [evaluation guide](../../README.md#optional-codex-cli-backend) and the committed dependency lock. Re-run into new ignored output directories; do not overwrite this archive. Model availability and backend revisions may change, and outputs may differ.

```bash
for model in gpt-6.1-sol gpt-6-astra; do
  LLM_BACKEND=codex CODEX_MODEL="$model" CODEX_REASONING_EFFORT=medium \
  MAX_DIAGNOSIS_RETRIES=0 LLM_TIMEOUT_SECONDS=600 \
  uv run --locked python -m evaluation.run_eval --cases case_001,case_002 --repeats 1
done
```

Regenerate the comparison plots from the saved summaries, without model calls:

```bash
uv run --locked --extra plots python -m evaluation.plot_comparison \
  --results evaluation/results/codex-comparison-2026-10-26/gpt-6.1-sol/eval_results.json \
  evaluation/results/codex-comparison-2026-10-26/gpt-6-astra/eval_results.json
```
