# Interrupted ultra pilot

This exploratory attempt used `gpt-6-astra`, `ultra` reasoning, a 300-second per-call timeout, and the original two-loopback diagnosis budget. Runtime source: `7342f6b`. The saved case_001 trace contains three diagnosis attempts, a Treatment timeout, and an end-to-end duration of 829.18 seconds. It is a partial stage failure, not a completed error-free pipeline.

The user reduced the scope during case_002, before that case was archived. Only case_001 has a finalized raw record; no five-case aggregate report was produced. The captured output is preserved unchanged.

This pilot is separate from the matched two-case, two-model medium-reasoning comparison. Its different settings and incomplete coverage do not support a model comparison. Timed-out calls have no final output or completed event stream; empty tool-use and usage fields for those calls are missing evidence.
