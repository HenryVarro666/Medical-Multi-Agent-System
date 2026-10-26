# Medical Multi-Agent System

[![Python CI](https://github.com/HenryVarro666/medical-multi-agent-system/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryVarro666/medical-multi-agent-system/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A Python research prototype that runs a synthetic patient narrative through five stages: intake, diagnosis, treatment, coding, and audit. LangGraph controls the workflow, and FastAPI exposes the pipeline and deterministic lookup tools.

The project demonstrates agent orchestration, retrieval traces, deterministic drug-interaction checks, and reproducible evaluation artifacts. It is not clinically validated, HIPAA certified, or intended for patient care. Use synthetic data only.

Adapted from [bcefghj/medical-multi-agent-system](https://github.com/bcefghj/medical-multi-agent-system), under the MIT license. See [attribution and project history](ATTRIBUTION.md).

## What runs today

- Four LLM stages produce structured intake, diagnosis, treatment, and coding outputs. A deterministic audit stage screens outputs with limited PHI regex patterns and marks deployment controls as unverified.
- Diagnosis retrieves candidates from a small, bundled symptom-to-disease map and includes the retrieval trace in its output. The default pipeline does not query Neo4j or a literature corpus.
- Treatment reconciles LLM suggestions with curated drug-interaction and allergy rules. These rules are illustrative and incomplete.
- ICD-10 lookup and simplified DRG grouping use bundled mappings. They do not replace a complete coding catalog or certified grouper.
- Evaluation records per-run outputs, scoring, errors, and latency for five synthetic vignettes.

## Quick start

Use Python 3.11 or newer; `.python-version` selects Python 3.12 for uv. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` to use the default OpenAI backend. Alternatively, point the application at an already running OpenAI-compatible model server:

```dotenv
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=local
OPENAI_MODEL=your-served-model
```

The model server is a separate dependency and is not launched by this repository. Keep it on a different port from the API. `LLM_DISABLE_THINKING` defaults to `false`; enable it only if the server supports `chat_template_kwargs.enable_thinking`.

Start the API:

```bash
uvicorn medical_multi_agent.api.main:app --host 127.0.0.1 --port 8000
```

Open [Swagger UI](http://127.0.0.1:8000/docs) or [ReDoc](http://127.0.0.1:8000/redoc). The health, ICD-10, and DDI endpoints work without an LLM. Full analysis requires a working configured backend. `/health` checks the API process, not model readiness.

For an installation using the committed dependency lockfile, use uv:

```bash
uv sync --extra dev --locked
uv run uvicorn medical_multi_agent.api.main:app --host 127.0.0.1 --port 8000
```

## Try the API

A deterministic interaction check:

```bash
curl -s http://127.0.0.1:8000/api/v1/clinical/ddi/check \
  -H 'Content-Type: application/json' \
  -d '{"new_drugs":["warfarin"],"current_drugs":["aspirin"]}'
```

Run a bundled synthetic vignette through the full pipeline:

```bash
python - <<'PY'
import json
from pathlib import Path
import httpx

case = json.loads(Path("data/sample_patients.json").read_text())[0]
response = httpx.post(
    "http://127.0.0.1:8000/api/v1/clinical/analyze",
    json={"patient_description": case["description"]},
    timeout=600,
)
response.raise_for_status()
print(json.dumps(response.json(), indent=2))
PY
```

Inspect the returned `errors` list as well as the HTTP status; individual agent failures can produce partial results. See the [API reference](docs/api.md).

## Workflow

```mermaid
flowchart LR
    Input["Synthetic narrative"] --> Intake
    Intake --> Diagnosis
    Map["Bundled symptom map"] --> Diagnosis
    Diagnosis -->|"Needs more information; retry budget remains"| Intake
    Diagnosis -->|"Ready or retry budget exhausted"| Treatment
    Rules["Curated DDI and allergy rules"] --> Treatment
    Treatment --> Coding
    Coding --> Audit["PHI screening and control checklist"]
    Audit --> Output["Structured result and errors"]
```

Retries reprocess the same narrative; the API does not collect new information from a user. Once the retry budget is exhausted, the current workflow proceeds with the available information. See [architecture and limitations](docs/architecture.md).

## Evaluation

### Recorded results

The original July 13, 2026 run used `Qwen/Qwen3-14B` through a local OpenAI-compatible endpoint. Five synthetic cases were run twice, producing ten recorded pipeline outputs. The original results remain displayed below and are preserved in the [full report](evaluation/results/historical-2026-07-13/eval_results.md), [machine-readable summary](evaluation/results/historical-2026-07-13/eval_results.json), and [raw outputs](evaluation/results/historical-2026-07-13/raw/).

[![Historical evaluation metrics: strict diagnosis alias matches 6/10, either-tier top-1 and top-3 matches 10/10, exact ICD-10 matches 5/10, category matches 10/10, and three structure/execution checks 10/10.](docs/assets/evaluation/historical-metrics.png)](docs/assets/evaluation/historical-metrics.svg)

[![Paired end-to-end latency for five synthetic cases over two repeats, with an overall mean of 28.22 seconds.](docs/assets/evaluation/historical-latency.png)](docs/assets/evaluation/historical-latency.svg)

The charts are generated from the preserved JSON. Click either chart for its vector version; see [figure regeneration](evaluation/README.md#readme-figures) for the command. Exact values and all ten raw runs remain below.

| Recorded measure | Original field | Result |
|---|---|---:|
| Top-1 diagnosis, strict alias tier | `diagnosis_top1_strict` | 6/10 (60%) |
| Top-1 diagnosis, either alias tier | `diagnosis_top1_any` | 10/10 (100%) |
| Top-3 diagnosis, either alias tier | `diagnosis_top3_any` | 10/10 (100%) |
| Exact ICD-10 match | `icd10_exact` | 5/10 (50%) |
| ICD-10 category match | `icd10_category` | 10/10 (100%) |
| Treatment structure complete | `treatment_ok` | 10/10 (100%) |
| Deterministic DDI check executed | `ddi_engine_checked` | 10/10 (100%) |
| Audit report produced | `audit_completed` | 10/10 (100%) |
| Mean pipeline latency | `mean_latency_seconds` | 28.22 s |

`treatment_ok` measures the presence of medications and a DDI execution stamp. `audit_completed` measures whether a report exists. These fields do not assess treatment safety or establish compliance.

<details>
<summary>Show all ten original per-case results</summary>

Each case links to its original raw pipeline output. “Plan” is the historical `treatment_ok` structural check; “DDI” and “Audit” indicate execution/report completion.

| Case | Expected diagnosis | Predicted diagnosis | Top-1 tier | Top-3 | Expected ICD | Predicted ICD | ICD exact | ICD category | Plan | DDI | Audit | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| [case_001 r1](evaluation/results/historical-2026-07-13/raw/case_001_r1.json) | Community-Acquired Pneumonia | Community-acquired pneumonia | strict | yes | J18.1 | J18.9 | no | yes | yes | yes | yes | 27.10 |
| [case_002 r1](evaluation/results/historical-2026-07-13/raw/case_002_r1.json) | ST-Elevation Myocardial Infarction (STEMI) | Acute myocardial infarction | compatible | yes | I21.0 | I21.9 | no | yes | yes | yes | yes | 29.16 |
| [case_003 r1](evaluation/results/historical-2026-07-13/raw/case_003_r1.json) | Hypothyroidism (Hashimoto's Thyroiditis) | Hypothyroidism | strict | yes | E03.9 | E03.9 | yes | yes | yes | yes | yes | 24.38 |
| [case_004 r1](evaluation/results/historical-2026-07-13/raw/case_004_r1.json) | Acute Decompensated Heart Failure | Heart Failure with Reduced Ejection Fraction (HFrEF) | compatible | yes | I50.9 | I50.21 | no | yes | yes | yes | yes | 35.92 |
| [case_005 r1](evaluation/results/historical-2026-07-13/raw/case_005_r1.json) | Acute Appendicitis | Appendicitis | strict | yes | K35.80 | K35.80 | yes | yes | yes | yes | yes | 24.29 |
| [case_001 r2](evaluation/results/historical-2026-07-13/raw/case_001_r2.json) | Community-Acquired Pneumonia | Community-acquired pneumonia | strict | yes | J18.1 | J18.9 | no | yes | yes | yes | yes | 27.48 |
| [case_002 r2](evaluation/results/historical-2026-07-13/raw/case_002_r2.json) | ST-Elevation Myocardial Infarction (STEMI) | Acute myocardial infarction | compatible | yes | I21.0 | I21.9 | no | yes | yes | yes | yes | 30.52 |
| [case_003 r2](evaluation/results/historical-2026-07-13/raw/case_003_r2.json) | Hypothyroidism (Hashimoto's Thyroiditis) | Hypothyroidism | strict | yes | E03.9 | E03.9 | yes | yes | yes | yes | yes | 26.83 |
| [case_004 r2](evaluation/results/historical-2026-07-13/raw/case_004_r2.json) | Acute Decompensated Heart Failure | Congestive Heart Failure (CHF) | compatible | yes | I50.9 | I50.9 | yes | yes | yes | yes | yes | 33.62 |
| [case_005 r2](evaluation/results/historical-2026-07-13/raw/case_005_r2.json) | Acute Appendicitis | Appendicitis | strict | yes | K35.80 | K35.80 | yes | yes | yes | yes | yes | 22.90 |

</details>

These are historical demo measurements, not results for the current refactor. Diagnosis scoring uses substring aliases; it does not assess clinical correctness. Repeats reuse the same five cases, labels have not been independently validated, and there is no baseline comparison or demonstrated GraphRAG improvement. The historical files retain their original wording and outputs for traceability.

### Run a new evaluation

After configuring a backend, create a new run:

```bash
python -m evaluation.run_eval --help
python -m evaluation.run_eval --repeats 2
python -m evaluation.run_eval --cases case_001 --output-dir /tmp/clinical-eval-new
```

New outputs default to an ignored, timestamped directory under `evaluation/results/runs/`. Historical results and nonempty output directories cannot be overwritten by the runner. See [evaluation methods and reproduction limits](evaluation/README.md).

## Development

```bash
python -m pytest
python -m ruff check .
```

The automated tests exercise the graph and agents with synthetic fixtures and mocked model calls; they also cover the API and evaluation runner. They do not require an LLM or clinical data. CI checks Python 3.11, 3.12, and 3.13, builds the package, and runs container smoke tests. A live evaluation is a separate check requiring a model backend.

```text
src/medical_multi_agent/   Agents, graph, API, models, configuration, services
data/                     Five synthetic vignette fixtures
tests/                    Offline regression tests
evaluation/               Scoring, runner, and historical artifacts
docs/                     Architecture and API reference
```

## Docker

```bash
cp .env.example .env
# Configure the backend in .env before full analysis.
docker compose up --build
```

Compose runs the API alone. For a model server running on the host, set `OPENAI_BASE_URL=http://host.docker.internal:8001/v1`. PostgreSQL and Redis are not used by the default pipeline. Neo4j connection helpers are optional and are not wired into diagnosis retrieval.

## Scope and limitations

The audit stage runs after LLM calls and does not redact the returned patient data. Remote model configuration sends narrative and intermediate data to the selected endpoint. This application has no authentication or durable audit storage. FHIR conversion helpers are experimental, are not invoked by the pipeline, and have not been validated against an external FHIR server.

The project is licensed under [MIT](LICENSE). Contributions should preserve the evidence boundaries described in [CONTRIBUTING.md](CONTRIBUTING.md).
