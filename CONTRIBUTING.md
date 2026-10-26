# Contributing

Use Python 3.11 or newer and install from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
```

For the committed dependency versions, use `uv sync --extra dev --locked` and run checks with `uv run`.

Keep changes focused and add regression tests for behavior changes. Tests should use synthetic inputs and mocked model calls. A model-backed evaluation is a separate, explicitly configured run; document the model, serving configuration, hardware, source revision, and limitations when reporting its results.

Preserve upstream attribution and the MIT license notice. Do not modify historical evaluation files. Write new runs to ignored directories under `evaluation/results/runs/`, or specify a new external output directory.

Do not commit `.env`, credentials, patient data, or identifiable medical records. Feature descriptions should distinguish implemented behavior, experimental helpers, and future ideas. Accuracy, safety, performance, integration, and compliance claims require evidence appropriate to the claim.

Include the problem, resulting behavior, and checks performed in a pull request. If a change alters scoring rules, describe how comparisons with historical measurements are affected.
