# Medical Multi-Agent System

[![Python CI](https://github.com/HenryVarro666/Medical-Multi-Agent-System/actions/workflows/ci.yml/badge.svg)](https://github.com/HenryVarro666/Medical-Multi-Agent-System/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A Python research prototype for a five-stage clinical workflow: Intake, Diagnosis, Treatment, Coding, and Audit. LangGraph orchestrates the Agents; FastAPI exposes analysis and deterministic lookup endpoints.

Use synthetic data only. The project is not clinically validated or HIPAA certified.

## Contents

- [Overview](#overview)
- [Data storage and retrieval](#data-storage-and-retrieval)
- [Quick start](#quick-start)
- [Model configuration](#model-configuration)
- [Try the API](#try-the-api)
- [Architecture and Agent workflow](#architecture)
- [End-to-end example](#end-to-end-example)
- [Evaluation](#evaluation)
- [Current model comparison](#current-model-comparison)
- [Historical results](#historical-results)
- [Docker](#docker)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Scope and attribution](#scope-and-attribution)
- [Current limitations](#current-limitations)
- [Upgrade and extension roadmap](#upgrade-and-extension-roadmap)

## Overview

| Component | Implemented behavior |
|---|---|
| Agent workflow | Four LLM stages followed by deterministic Audit, with bounded diagnosis retries |
| Retrieval | A small bundled symptom map; candidates and retrieval traces enter Diagnosis |
| Treatment checks | Curated DDI/allergy rules reconcile the model's proposal |
| Offline API tools | Health, ICD-10 search/lookup, simplified DRG mapping, and DDI checks |
| Evaluation | Five synthetic fixtures, preserved historical runs, and reproducible figures |

The API and offline tools require no model server, GPU, or database. Full analysis requires a configured OpenAI-compatible backend or a locally installed, authenticated Codex CLI. Neo4j/FHIR helpers are outside the default workflow.

## Data storage and retrieval

**The default pipeline uses in-memory Python objects and explicit JSON files. It does not use a running database or vector index.**

| Data | Current location | Lifetime / persistence |
|---|---|---|
| Synthetic input fixtures | [data/sample_patients.json](data/sample_patients.json) | Five version-controlled examples |
| Patient information and Agent outputs | [ClinicalState](src/medical_multi_agent/graph/state.py) | Memory during one invocation; the API returns the result without saving a patient/session record |
| Retrieval knowledge | [graphrag_service.py](src/medical_multi_agent/services/graphrag_service.py) | Python dictionaries with 10 symptom keys and 15 disease/code hints; exact normalized symptom overlap |
| ICD-10 / DRG and medication rules | [ICD/DRG tables](src/medical_multi_agent/services/icd10_service.py), [DDI/allergy rules](src/medical_multi_agent/services/drug_interaction.py) | Small bundled Python mappings; not a complete clinical knowledge base |
| Evaluation outputs and model-call traces | [evaluation/results/](evaluation/results/) | Runner explicitly writes raw JSON and JSON/Markdown summaries; local `runs/` are Git-ignored, reviewed archives are committed |
| Audit result | Returned inside the pipeline result | No durable audit database in the default API; evaluation archives retain a copy |

Checkpoint retention is disabled by default. Library callers can pass a checkpointer to `build_clinical_pipeline(checkpointer=...)`; its persistence depends on the supplied implementation. JSON evaluation archives are separate from API session storage.

### What the Neo4j code currently does

[Neo4j uses a property graph model](https://neo4j.com/docs/getting-started/appendix/graphdb-concepts/) and also supports [vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/). Graph traversal and embedding similarity search are different retrieval mechanisms.

This project currently uses neither through Neo4j. `get_graphrag_service()` explicitly sets `use_neo4j=False`; `find_diseases_by_symptoms()` always reads the bundled dictionaries. Installing the optional `neo4j` extra or setting `NEO4J_*` variables does not switch Diagnosis to a database. The asynchronous `connect()` / `query_neo4j()` helpers must be integrated explicitly. FHIR helpers are also outside the default workflow.

The legacy component name `GraphRAGService` describes a retrieval hint service here: there is no ingested medical graph, embedding model, vector index, document retrieval, or multi-hop graph reasoning. See [backend alternatives and compatibility](#storage-and-retrieval-alternatives) for proposed extensions.

## Quick start

### 1. Clone and install

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). The commands below use macOS/Linux shell syntax. This public repository can be cloned without GitHub credentials.

```bash
git clone https://github.com/HenryVarro666/Medical-Multi-Agent-System.git
cd Medical-Multi-Agent-System
uv python install 3.12
uv sync --locked --managed-python
cp .env.example .env
```

The managed Python path avoids dependence on the system interpreter. uv creates `.venv` and installs the committed dependency versions. CI tests Python 3.11, 3.12, and 3.13; Python 3.12 is the default here.

For PowerShell, use `Copy-Item .env.example .env` for the last step. Keep all later commands in the repository root. The uv commands do not require virtual-environment activation.

<details>
<summary>Alternative: install with pip and an existing Python 3.12</summary>

After cloning and entering the repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell. Use `Copy-Item` to create `.env`.

For this installation path, replace `uv run --locked python` with `python` in later examples. Keep the venv active in each terminal. pip resolves compatible version ranges; it does not use `uv.lock`.

</details>

### 2. Start the API

In terminal 1, leave the server running:

```bash
uv run --locked python -m uvicorn medical_multi_agent.api.main:app --host 127.0.0.1 --port 8000
```

With pip, run the same command without `uv run --locked` in the activated venv. You can start the API with the unchanged `.env` template; credentials are needed only for full analysis.

### 3. Verify the offline endpoints

In terminal 2, run:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/api/v1/clinical/icd10/J18.9
curl --fail http://127.0.0.1:8000/api/v1/clinical/ddi/check \
  -H 'Content-Type: application/json' \
  -d '{"new_drugs":["warfarin"],"current_drugs":["aspirin"]}'
```

All three requests should return HTTP 200. The DDI example should report `has_major_interaction: true`. Open [Swagger UI](http://127.0.0.1:8000/docs) to inspect and try the endpoints.

`/health` reports process liveness. It does not test model readiness. Continue with model configuration before calling `/clinical/analyze`.

## Model configuration

Edit `.env` in the repository root and restart the API after changes. Choose one backend; this repository does not download model weights or launch a model server.

### Hosted OpenAI backend

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

### Existing local OpenAI-compatible backend

```dotenv
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=local
OPENAI_MODEL=your-served-model
```

The configured model name must match the server. Keep the model server on a different port from the API. In Docker, use `host.docker.internal` for a backend running on the host.

### Codex CLI with a ChatGPT subscription

Install the [Codex CLI](https://learn.chatgpt.com/docs/cli), then run `codex login` and choose ChatGPT sign-in. Confirm with `codex login status`. Configure `.env`:

```dotenv
LLM_BACKEND=codex
CODEX_MODEL=gpt-6.1-sol
CODEX_REASONING_EFFORT=medium
LLM_TIMEOUT_SECONDS=600
MAX_DIAGNOSIS_RETRIES=0
```

Use a model available to your account. This adapter is tested with Codex CLI `0.163.0`; the model and effort above follow this project's selected run configuration. Saved CLI authentication is reused; this backend does not require `OPENAI_API_KEY`. Each Agent call starts a fresh ephemeral CLI process outside the repository. The original system prompt becomes the CLI instructions file and the original human prompt goes through stdin. CLI context and decoding differ from Chat Completions; the original temperature settings are not applied. See [evaluation protocol](evaluation/README.md) and [OpenAI's non-interactive CLI documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `openai` | `openai` for compatible HTTP endpoints; `codex` for local CLI calls |
| `CODEX_MODEL` | Empty | Required for Codex; explicitly selects an account-accessible model |
| `CODEX_REASONING_EFFORT` | `medium` | Reasoning effort passed explicitly to Codex |
| `CODEX_CLI_PATH` | `codex` | CLI executable name or path |
| `OPENAI_API_KEY` | Empty | Required for a remote HTTP backend; `local` is a placeholder for a local server |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model identifier sent to the selected endpoint |
| `OPENAI_BASE_URL` | Empty | OpenAI by default; set the full compatible endpoint for a local/custom server |
| `LLM_TIMEOUT_SECONDS` | `180` | Timeout for each model call |
| `LLM_DISABLE_THINKING` | `false` | Opt-in `chat_template_kwargs.enable_thinking=false`; enable only on servers that support it |
| `MAX_DIAGNOSIS_RETRIES` | `2` | Diagnosis-to-Intake loopbacks, in addition to the initial attempt |

See [.env.example](.env.example) for optional Neo4j/FHIR helper settings. No external service is needed for the default offline endpoints.

## Try the API

After configuring a backend and restarting the API, run this from the repository root in terminal 2:

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
import httpx
from medical_multi_agent.config.settings import get_settings

case = json.loads(Path("data/sample_patients.json").read_text())[0]
settings = get_settings()
max_calls = 4 + 2 * settings.max_diagnosis_retries
attempts_per_call = 1 if settings.llm_backend == "codex" else 2
response = httpx.post(
    "http://127.0.0.1:8000/api/v1/clinical/analyze",
    json={"patient_description": case["description"]},
    timeout=settings.llm_timeout_seconds * max_calls * attempts_per_call + 60,
)
response.raise_for_status()
result = response.json()
print(json.dumps(result, indent=2))
if result["errors"]:
    raise SystemExit("One or more stages failed; inspect the errors above.")
PY
```

For pip, use `python - <<'PY'` in the activated venv. Agent failures can return HTTP 200 with partial sections and a nonempty `errors` list, so inspect both the status and response body.

The client timeout allows all model stages and configured retries. Reasoning-heavy Codex calls can take several minutes per stage; the evaluation runner invokes the graph directly and does not add an HTTP client timeout.

See the [API reference](docs/api.md) for schemas and all endpoints. The [archived walkthrough](#end-to-end-example) shows a complete recorded run without requiring a model connection.

## Architecture

### System diagram

[![System architecture: FastAPI invokes a sequential LangGraph workflow with ClinicalState, an OpenAI-compatible backend, bundled retrieval and treatment rules, plus standalone ICD-10/DRG tools.](docs/assets/architecture/system-architecture.png)](docs/assets/architecture/system-architecture.svg)

The four LLM stages share the client factory. Audit runs deterministic checks after model calls. State lasts for one invocation by default; checkpoint retention is opt-in.

<a id="workflow"></a>

### Agent workflow

[![Five-Agent workflow: Intake, Diagnosis, Treatment, Coding, and Audit execute in order, with a bounded Diagnosis-to-Intake retry.](docs/assets/architecture/agent-workflow.png)](docs/assets/architecture/agent-workflow.svg)

Retries reprocess the same narrative. Once the retry budget is exhausted, the workflow proceeds with the available information. See [architecture details and figure sources](docs/architecture.md).

<details>
<summary>Compact workflow in Mermaid</summary>

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

</details>

## End-to-end example

This walkthrough preserves `case_001`, repeat 1, recorded on July 13, 2026 with `Qwen/Qwen3-14B`. It comes from a direct graph invocation by the evaluation harness; it is a historical result, not a new execution. The [current two-model comparison](#current-model-comparison) provides separate, newly executed traces for the same case.

[![Historical case_001: input, five key Agent outputs, and a 27.10-second final result with an exact ICD-10 mismatch.](docs/assets/examples/case-001-walkthrough.png)](docs/assets/examples/case-001-walkthrough.svg)

The [raw trace](evaluation/results/historical-2026-07-13/raw/case_001_r1.json) contains every recorded field. The figure shows the single observed path; complete stage JSON remains expandable below.

### Input

> 45-year-old male presenting with fever (39.2°C) for 3 days, productive cough with yellow sputum, and right-sided chest pain that worsens with deep breathing. History of type 2 diabetes (diagnosed 5 years ago) and hypertension. Current medications: metformin 500mg twice daily, lisinopril 10mg daily. Allergies: penicillin (causes rash). Vital signs: BP 130/85, HR 102, RR 22, SpO2 94% on room air. Labs: WBC 15,000/μL (elevated), CRP 85 mg/L (elevated), procalcitonin 2.1 ng/mL (elevated). Chest X-ray shows right lower lobe consolidation.

The API accepts this narrative as `patient_description` and maps it to `raw_input`. Fixture labels `Community-Acquired Pneumonia` and `J18.1` are used for scoring and are not included in the model input.

### Recorded Agent handoffs

| Agent | State field | Key recorded output |
|---|---|---|
| Intake | `patient_info` | Age 45, male; structured symptoms/labs, two medications, penicillin allergy |
| Diagnosis | `diagnosis` | Community-acquired pneumonia; model confidence 0.85; five retrieval candidates |
| Treatment | `treatment_plan` | Two medication proposals; DDI executed; 0 curated-rule matches and 4 LLM interaction entries |
| Coding | `coding_result` | Model-proposed primary code `J18.9` and DRG `193` |
| Audit | `audit_result` | Historical `hipaa_compliant: false`, `name` pattern flagged; unsupported masking/control claims |

Medication, interaction, coding, confidence, and reference fields are unverified model output. Legacy audit claims do not establish masking or compliance. These archived values explain the workflow; they are not clinical instructions.

### Final recorded result

| Item | Recorded result |
|---|---|
| Primary diagnosis | `Community-acquired pneumonia` |
| Diagnosis alias tier | `strict` |
| Fixture / predicted ICD-10 | `J18.1` / `J18.9` |
| Exact ICD-10 / category match | `false` / `true` |
| Treatment proposal | `Levofloxacin`, `Acetaminophen`; unverified model output |
| Deterministic treatment checks | Executed; 0 curated-rule interaction matches, 0 rule-based allergy alerts |
| Historical audit | `hipaa_compliant: false`; `name` flagged; legacy masking/control claims are unsupported |
| Diagnosis attempts | `1` |
| Stored pipeline `errors` / `harness_error` | `null` / `null` (no errors recorded) |
| End-to-end latency | **27.10 seconds** |

The run completed with all five sections present. Exact ICD-10 matching failed while the category matched. The historical audit flagged a name pattern; the current audit no longer asserts masking or verified deployment controls.

### Complete Agent outputs

<details>
<summary>1. Intake — complete patient_info JSON</summary>

Intake converts the narrative to structured patient information. The stored object becomes input to Diagnosis.

```json
{
  "patient_id": null,
  "name": "Unknown",
  "age": 45,
  "gender": "male",
  "chief_complaint": "fever, productive cough with yellow sputum, and right-sided chest pain that worsens with deep breathing",
  "symptoms": [
    {
      "name": "fever",
      "duration_days": 3,
      "severity": "severe",
      "description": "temperature of 39.2°C"
    },
    {
      "name": "productive cough",
      "duration_days": 3,
      "severity": "moderate",
      "description": "with yellow sputum"
    },
    {
      "name": "right-sided chest pain",
      "duration_days": 3,
      "severity": "moderate",
      "description": "worsens with deep breathing"
    }
  ],
  "medical_history": [
    "type 2 diabetes",
    "hypertension"
  ],
  "family_history": [],
  "allergies": [
    {
      "substance": "penicillin",
      "reaction": "causes rash",
      "severity": "mild"
    }
  ],
  "current_medications": [
    {
      "name": "metformin",
      "dosage": "500mg",
      "frequency": "twice daily",
      "start_date": null
    },
    {
      "name": "lisinopril",
      "dosage": "10mg",
      "frequency": "daily",
      "start_date": null
    }
  ],
  "vital_signs": {
    "temperature": 39.2,
    "heart_rate": 102,
    "blood_pressure_systolic": 130,
    "blood_pressure_diastolic": 85,
    "respiratory_rate": 22,
    "oxygen_saturation": 94.0
  },
  "lab_results": [
    {
      "test_name": "WBC",
      "value": "15,000/μL",
      "unit": "/μL",
      "reference_range": "4,500–11,000",
      "is_abnormal": true
    },
    {
      "test_name": "CRP",
      "value": "85 mg/L",
      "unit": "mg/L",
      "reference_range": "< 10",
      "is_abnormal": true
    },
    {
      "test_name": "procalcitonin",
      "value": "2.1 ng/mL",
      "unit": "ng/mL",
      "reference_range": "< 0.5",
      "is_abnormal": true
    }
  ],
  "raw_input": null,
  "created_at": "2026-07-12T19:46:04.668905"
}
```

</details>

<details>
<summary>2. Diagnosis — complete diagnosis JSON</summary>

Diagnosis uses patient information and bundled retrieval hints. Confidence is model-reported, not calibrated accuracy; source-name strings are not verified citations.

```json
{
  "primary_diagnosis": {
    "disease_name": "Community-acquired pneumonia",
    "icd10_hint": "J18.9",
    "confidence": 0.85,
    "evidence": [
      "productive cough with yellow sputum",
      "right-sided chest pain that worsens with deep breathing (pleuritic chest pain)",
      "fever of 39.2°C",
      "elevated WBC (15,000/μL), elevated CRP (85 mg/L), and elevated procalcitonin (2.1 ng/mL) indicating bacterial infection"
    ],
    "reasoning": "The combination of fever, productive cough with yellow sputum, pleuritic chest pain, and elevated inflammatory markers (WBC, CRP, procalcitonin) strongly suggests a bacterial infection, most likely community-acquired pneumonia. The presence of pleuritic chest pain and yellow sputum further supports this diagnosis."
  },
  "differential_list": [
    {
      "disease_name": "Sepsis",
      "icd10_hint": "A41.9",
      "confidence": 0.6,
      "evidence": [
        "fever",
        "elevated WBC, CRP, and procalcitonin"
      ],
      "reasoning": "Sepsis can present with fever and systemic inflammatory markers. However, the absence of systemic signs such as hypotension, tachycardia beyond what is already present, or altered mental status makes sepsis less likely as the primary diagnosis."
    },
    {
      "disease_name": "Influenza",
      "icd10_hint": "J11.1",
      "confidence": 0.5,
      "evidence": [
        "fever"
      ],
      "reasoning": "Influenza can cause fever and cough, but it typically does not present with pleuritic chest pain or such markedly elevated inflammatory markers. Additionally, influenza is less likely to cause such a high procalcitonin level, which is more indicative of bacterial infection."
    },
    {
      "disease_name": "Pneumothorax",
      "icd10_hint": "J93.1",
      "confidence": 0.4,
      "evidence": [
        "right-sided chest pain that worsens with deep breathing"
      ],
      "reasoning": "Pneumothorax can cause pleuritic chest pain, but it is not typically associated with fever, productive cough, or elevated inflammatory markers. A chest X-ray would be necessary to confirm or rule out this diagnosis."
    }
  ],
  "recommended_tests": [
    "Chest X-ray to confirm pneumonia or other pulmonary pathology",
    "Sputum culture and sensitivity to identify the causative organism",
    "Blood culture to rule out sepsis",
    "Chest CT scan if the chest X-ray is inconclusive"
  ],
  "clinical_notes": "The patient presents with signs and symptoms consistent with community-acquired pneumonia, supported by elevated inflammatory markers. Sepsis and influenza are considered differentials, but the clinical picture is more consistent with a bacterial infection. Further diagnostic testing is recommended to confirm the diagnosis and guide treatment.",
  "knowledge_sources": [
    "UpToDate: Community-Acquired Pneumonia",
    "CDC: Influenza and Sepsis Guidelines",
    "ICD-10-CM Diagnosis Code Lookup",
    "GraphRAG knowledge graph (symptom->disease retrieval)"
  ],
  "knowledge_graph_candidates": [
    {
      "disease": "Influenza",
      "symptom_match_count": 1,
      "icd10_code": "J11.1",
      "icd10_description": "Influenza with other respiratory manifestations"
    },
    {
      "disease": "Pneumonia",
      "symptom_match_count": 1,
      "icd10_code": "J18.9",
      "icd10_description": "Pneumonia, unspecified organism"
    },
    {
      "disease": "COVID-19",
      "symptom_match_count": 1,
      "icd10_code": "U07.1",
      "icd10_description": "COVID-19, virus identified"
    },
    {
      "disease": "Sepsis",
      "symptom_match_count": 1,
      "icd10_code": "A41.9",
      "icd10_description": "Sepsis, unspecified organism"
    },
    {
      "disease": "Malaria",
      "symptom_match_count": 1,
      "icd10_code": "",
      "icd10_description": ""
    }
  ]
}
```

</details>

<details>
<summary>3. Treatment — complete treatment_plan JSON</summary>

The model proposes medications and four interaction descriptions. The deterministic stamp records zero curated-rule matches; it does not verify the model text or establish medication safety.

```json
{
  "diagnosis_addressed": "Community-acquired pneumonia",
  "medications": [
    {
      "drug_name": "Levofloxacin",
      "generic_name": "Levofloxacin",
      "dosage": "500mg",
      "route": "oral",
      "frequency": "once daily",
      "duration": "7 days",
      "contraindications": [
        "hypersensitivity to fluoroquinolones",
        "tendinitis or tendon rupture risk in elderly patients"
      ],
      "side_effects": [
        "nausea",
        "headache",
        "dizziness",
        "photosensitivity",
        "gastrointestinal upset"
      ]
    },
    {
      "drug_name": "Acetaminophen",
      "generic_name": "Acetaminophen",
      "dosage": "500mg",
      "route": "oral",
      "frequency": "every 4–6 hours as needed",
      "duration": "as needed for fever",
      "contraindications": [
        "severe liver disease",
        "alcoholism"
      ],
      "side_effects": [
        "nausea",
        "rash",
        "hepatotoxicity (rare)"
      ]
    }
  ],
  "drug_interactions": [
    {
      "drug_a": "Levofloxacin",
      "drug_b": "Lisinopril",
      "severity": "minor",
      "description": "Levofloxacin may increase the risk of hypoglycemia in patients taking ACE inhibitors like lisinopril, though this is rare.",
      "recommendation": "Monitor for signs of hypoglycemia and adjust lisinopril dosage if necessary."
    },
    {
      "drug_a": "Levofloxacin",
      "drug_b": "Metformin",
      "severity": "minor",
      "description": "Levofloxacin may increase the risk of lactic acidosis in patients taking metformin, though this is rare.",
      "recommendation": "Monitor for signs of lactic acidosis and consider alternative antibiotics if risk is high."
    },
    {
      "drug_a": "Acetaminophen",
      "drug_b": "Lisinopril",
      "severity": "none",
      "description": "No significant interaction between acetaminophen and lisinopril.",
      "recommendation": "No action required."
    },
    {
      "drug_a": "Acetaminophen",
      "drug_b": "Metformin",
      "severity": "none",
      "description": "No significant interaction between acetaminophen and metformin.",
      "recommendation": "No action required."
    }
  ],
  "non_drug_treatments": [
    "Hydration with oral fluids",
    "Chest physiotherapy to aid sputum clearance"
  ],
  "lifestyle_recommendations": [
    "Rest and avoid strenuous activity",
    "Maintain good hand hygiene",
    "Avoid smoking and secondhand smoke"
  ],
  "follow_up_plan": "Follow up in 48–72 hours to assess clinical improvement. If no improvement, re-evaluate for possible antibiotic resistance or complications. Repeat chest X-ray if symptoms persist.",
  "warnings": [
    "Monitor for signs of antibiotic resistance (e.g., worsening symptoms, new fever), allergic reactions to levofloxacin, and signs of lactic acidosis (e.g., nausea, abdominal pain, weakness). Avoid sun exposure while on levofloxacin due to photosensitivity risk."
  ],
  "evidence_references": [
    "UpToDate: Community-Acquired Pneumonia",
    "CDC Guidelines for the Treatment of Community-Acquired Pneumonia",
    "American Thoracic Society (ATS) and Infectious Diseases Society of America (IDSA) Guidelines"
  ],
  "allergy_alerts": [],
  "ddi_engine": {
    "checked": true,
    "proposed_drugs_checked": [
      "Levofloxacin",
      "Acetaminophen"
    ],
    "current_medications_checked": [
      "metformin",
      "lisinopril"
    ],
    "allergies_checked": [
      "penicillin"
    ],
    "engine_flagged_interactions": 0,
    "engine_flagged_allergy_alerts": 0
  }
}
```

</details>

<details>
<summary>4. Coding — complete coding_result JSON</summary>

These are model-generated code and DRG proposals. The primary code is J18.9, compared with fixture J18.1 only during evaluation.

```json
{
  "primary_icd10": {
    "code": "J18.9",
    "description": "Pneumonia, unspecified organism",
    "confidence": 0.92,
    "category": "Infectious and parasitic diseases"
  },
  "secondary_icd10_codes": [
    {
      "code": "R07.0",
      "description": "Pleuritic chest pain",
      "confidence": 0.85,
      "category": "Symptoms, signs, and abnormal clinical and laboratory findings, not elsewhere classified"
    },
    {
      "code": "R74.0",
      "description": "Elevated white blood cell count",
      "confidence": 0.8,
      "category": "Symptoms, signs, and abnormal clinical and laboratory findings, not elsewhere classified"
    },
    {
      "code": "R79.0",
      "description": "Elevated C-reactive protein",
      "confidence": 0.75,
      "category": "Symptoms, signs, and abnormal clinical and laboratory findings, not elsewhere classified"
    }
  ],
  "drg_group": {
    "drg_code": "193",
    "description": "Pneumonia, unspecified, with major complications or comorbidities",
    "weight": 1.2,
    "mean_los": 4.5
  },
  "coding_notes": "The primary diagnosis of community-acquired pneumonia is supported by clinical symptoms (productive cough, pleuritic chest pain, fever) and laboratory findings (elevated WBC, CRP, and procalcitonin). J18.9 is the most specific ICD-10-CM code for unspecified pneumonia. Secondary codes include pleuritic chest pain (R07.0), elevated WBC (R74.0), and elevated CRP (R79.0). DRG 193 is appropriate for pneumonia with complications or comorbidities, with a realistic weight and mean length of stay.",
  "coding_confidence": 0.9
}
```

</details>

<details>
<summary>5. Audit — complete historical audit_result JSON</summary>

The old code claimed masking and verified deployment controls without establishing either. This JSON preserves the old output; current code marks controls unverified and does not claim masking.

```json
{
  "hipaa_compliant": false,
  "compliance_checks": [
    {
      "check_name": "phi_scan",
      "passed": false,
      "detail": "Found 1 PHI types: name"
    },
    {
      "check_name": "data_encryption_at_rest",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "data_encryption_in_transit",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "access_control_rbac",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "audit_logging",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "minimum_necessary_rule",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "breach_notification_ready",
      "passed": true,
      "detail": "Verified"
    },
    {
      "check_name": "data_retention_policy",
      "passed": true,
      "detail": "Verified"
    }
  ],
  "phi_fields_found": [
    "name"
  ],
  "phi_fields_masked": [
    "name"
  ],
  "audit_trail": [
    {
      "timestamp": "2026-07-13T00:46:25.598644+00:00",
      "user_id": "system",
      "action": "phi_scan",
      "resource_type": "pipeline_output",
      "resource_id": "",
      "detail": "Scanned 4 sections",
      "outcome": "success",
      "ip_address": ""
    },
    {
      "timestamp": "2026-07-13T00:46:25.602111+00:00",
      "user_id": "system",
      "action": "data_masking",
      "resource_type": "pipeline_output",
      "resource_id": "",
      "detail": "Masked 1 PHI types",
      "outcome": "success",
      "ip_address": ""
    },
    {
      "timestamp": "2026-07-13T00:46:25.602255+00:00",
      "user_id": "system",
      "action": "compliance_assessment",
      "resource_type": "pipeline",
      "resource_id": "",
      "detail": "Overall: NEEDS_REVIEW, risk=medium",
      "outcome": "success",
      "ip_address": ""
    }
  ],
  "recommendations": [
    "Ensure all PHI is masked before external transmission",
    "Review failed compliance checks and remediate",
    "Maintain audit logs for minimum 6 years per HIPAA requirements"
  ],
  "overall_risk_level": "medium"
}
```

</details>

## Evaluation

### Current model comparison

On October 26, 2026, two original fixtures (`case_001` and `case_002`) were run once per model through the five-stage pipeline using the signed-in Codex CLI. Both models returned all five Agent sections for both cases, with **no pipeline or harness errors**. Four pipeline outputs contain 16 completed CLI invocations; Audit is deterministic.

The matched protocol uses **medium reasoning**, a **600-second per-call timeout**, and **zero diagnosis loopbacks**. Fixed fixtures cannot answer follow-up questions, so each case follows the single-pass path even if Diagnosis requests more information. Original stage prompts, retrieval/rule code, source revision, and dependency lock are shared. The two model runs were sequential in a clean clone with Python 3.12.13 and Codex CLI 0.163.0.

[![Two models compared on eight fixture-label and stage-execution checks.](docs/assets/evaluation/codex-comparison-2026-10-26-metrics.png)](docs/assets/evaluation/codex-comparison-2026-10-26-metrics.svg)

[![Four pipeline timings: gpt-6.1-sol averages 156.91 seconds and gpt-6-astra averages 206.44 seconds.](docs/assets/evaluation/codex-comparison-2026-10-26-latency.png)](docs/assets/evaluation/codex-comparison-2026-10-26-latency.svg)

| Recorded check | `gpt-6.1-sol` | `gpt-6-astra` |
|---|---:|---:|
| Error-free five-stage outputs | 2/2 | 2/2 |
| Top-1 diagnosis, strict alias | 2/2 (100%) | 2/2 (100%) |
| Top-1 diagnosis, either alias | 2/2 (100%) | 2/2 (100%) |
| Top-3 diagnosis, either alias | 2/2 (100%) | 2/2 (100%) |
| Exact ICD-10 label match | 0/2 (0%) | 0/2 (0%) |
| ICD-10 category match | 2/2 (100%) | 2/2 (100%) |
| Treatment structure complete | 2/2 (100%) | 2/2 (100%) |
| DDI check executed | 2/2 (100%) | 2/2 (100%) |
| Audit report produced | 2/2 (100%) | 2/2 (100%) |
| Mean pipeline elapsed time | 156.91 s | 206.44 s |

Both models produced `J18.9` for the pneumonia case and `I21.09` for the STEMI case; the original labels are `J18.1` and `I21.0`. These are exact-label mismatches with matching code categories. The evaluation does not adjudicate which code is clinically correct.

**Why is exact matching 0%?** [`icd10_scores()`](evaluation/run_eval.py) trims outer whitespace, uppercases both strings, and compares the complete codes. `J18.1 != J18.9` and `I21.0 != I21.09`, so both models have **0 exact matches out of 2**. The category check compares only the portion before the decimal (`J18` / `I21`), giving 2/2; it does not validate code existence or clinical equivalence.

There is also a label-quality problem. The [CMS FY 2027 code files](https://www.cms.gov/files/zip/2027-code-descriptions-tabular-order.zip) list `I21.0` as a parent/header with more specific children including `I21.09`; it is absent from the complete-code list. Yet the Coding prompt asks for the most specific code. The fixtures do not specify a coding-year or encounter-setting contract, and their labels have not been independently adjudicated. This structural mismatch does not establish that `I21.09` is the correct code for this vignette.

For pneumonia, the retrieval map supplies `J18.9`, which both models retain, while the fixture expects `J18.1`. That suggests a possible anchoring effect, not a proven cause; an ablation is needed. **Keep the recorded 0% as fixture-label agreement, not clinical coding accuracy.** A future, versioned evaluation should review the labels, validate catalog membership and specificity, and report expert-adjudicated acceptability separately. The bundled validator checks only a small demonstration subset, so simply wiring it into Coding would not provide authoritative validation.

| Complete raw trace | Model | Expected → generated ICD-10 | More information requested | Elapsed time |
|---|---|---|---|---:|
| [case_001](evaluation/results/codex-comparison-2026-10-26/gpt-6.1-sol/raw/case_001_r1.json) | `gpt-6.1-sol` | `J18.1` → `J18.9` | no | 144.33 s |
| [case_001](evaluation/results/codex-comparison-2026-10-26/gpt-6-astra/raw/case_001_r1.json) | `gpt-6-astra` | `J18.1` → `J18.9` | yes | 197.84 s |
| [case_002](evaluation/results/codex-comparison-2026-10-26/gpt-6.1-sol/raw/case_002_r1.json) | `gpt-6.1-sol` | `I21.0` → `I21.09` | no | 169.49 s |
| [case_002](evaluation/results/codex-comparison-2026-10-26/gpt-6-astra/raw/case_002_r1.json) | `gpt-6-astra` | `I21.0` → `I21.09` | no | 215.04 s |

`gpt-6-astra` requested more information on `case_001`; the other three outputs did not. The raw records include the input narrative, submitted prompts, every model response, deterministic enrichment, five final Agent sections, and scoring. Returning all sections establishes execution completion, not clinical completeness, treatment safety, or compliance.

`gpt-6.1-sol` was faster on these two observations. With two shared synthetic cases and one repeat, this does not establish a general model ranking. Timing includes CLI startup, transport, caching/queueing, and inference. Temperature is unsupported by this transport; backend model revisions are not exposed. Historical Qwen runs below have a different protocol and denominator.

See the [comparison archive and reproduction notes](evaluation/results/codex-comparison-2026-10-26/README.md), [Sol report](evaluation/results/codex-comparison-2026-10-26/gpt-6.1-sol/eval_results.md), and [Astra report](evaluation/results/codex-comparison-2026-10-26/gpt-6-astra/eval_results.md). An earlier `ultra` pilot hit a Treatment timeout; its [unchanged raw record and interruption notes](evaluation/results/codex-comparison-2026-10-26/exploratory-astra-ultra/README.md) are retained separately and excluded from this comparison.

### Historical results

The July 13, 2026 run used `Qwen/Qwen3-14B`: five synthetic cases, two repeats, ten pipeline outputs. The original [report](evaluation/results/historical-2026-07-13/eval_results.md) and JSON artifacts are preserved.

[![Eight historical checks: diagnosis alias matching, ICD-10 matching, treatment structure, DDI execution, and audit completion.](docs/assets/evaluation/historical-metrics.png)](docs/assets/evaluation/historical-metrics.svg)

[![Two recorded latencies per case, with a 28.22-second overall mean.](docs/assets/evaluation/historical-latency.png)](docs/assets/evaluation/historical-latency.svg)

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

`treatment_ok` checks medications and a DDI execution stamp. `audit_completed` checks that a report exists. Neither measures treatment safety or compliance.

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

These are historical demo measurements. Diagnosis scoring uses substring aliases, not clinical adjudication. Repeated runs reuse five cases; labels have not been independently validated. No baseline or retrieval improvement is established.

### Run a new evaluation

With a configured backend, run from the repository root:

```bash
uv run --locked python -m evaluation.run_eval --help
uv run --locked python -m evaluation.run_eval --cases case_001 --repeats 1
uv run --locked python -m evaluation.run_eval --repeats 2
```

New runs use separate directories under `evaluation/results/runs/`. The runner rejects overwriting historical or nonempty result directories. Model output may differ from the archive. See [methods and scoring limits](evaluation/README.md).

## Docker

Use the `.env` created in quick start. If you skipped that step, copy `.env.example` to `.env` once, then configure it. From the repository root:

Stop the API in terminal 1 with Ctrl+C before starting Compose, or choose a different host port to avoid a conflict on port 8000.

```bash
docker compose up --build --wait
```

Docker Compose starts only the API. Full analysis still requires a configured model backend.

The supplied container supports the OpenAI-compatible HTTP backend. Run the Codex backend on the host; the image does not include the CLI or your saved login.

Try the same health and offline API requests from [quick start](#3-verify-the-offline-endpoints). For a host model server, use `OPENAI_BASE_URL=http://host.docker.internal:8001/v1`. `localhost` inside the container means the container itself.

```bash
docker compose logs -f api
docker compose down
```

## Development

### Tests and package builds

```bash
uv sync --locked --managed-python --extra dev
uv run --locked --extra dev pytest
uv run --locked --extra dev ruff check .
uv build
```

For pip, install `python -m pip install -e ".[dev]"`, then run `python -m pytest` and `python -m ruff check .` in the activated venv.

Tests use synthetic fixtures and mocked model calls. CI checks Python 3.11–3.13, pip installation, package builds, figure generation, and Docker startup. Live model inference requires a separately configured backend.

### Regenerate the figures

```bash
uv run --locked --extra plots python -m evaluation.plot_results
uv run --locked --extra plots python -m evaluation.plot_comparison \
  --results evaluation/results/codex-comparison-2026-10-26/gpt-6.1-sol/eval_results.json \
  evaluation/results/codex-comparison-2026-10-26/gpt-6-astra/eval_results.json
uv run --locked --extra plots python -m evaluation.draw_architecture
uv run --locked --extra plots python -m evaluation.draw_case
```

These commands read local data and make no model calls. The optional `plots` dependency is needed only for regeneration; committed PNG/SVG files display without it. See [figure details](evaluation/README.md#readme-figures).

### Project layout

```text
src/medical_multi_agent/   Agents, graph, API, models, configuration, services
data/                     Five synthetic vignette fixtures
tests/                    Offline regression tests
evaluation/               Runner, scoring, figure generators, historical artifacts
docs/                     Architecture, API reference, and image assets
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for change and evidence requirements.

## Troubleshooting

| Symptom | Check |
|---|---|
| Clone fails or asks for credentials | Use the public HTTPS clone URL above; check network access and local Git/proxy settings |
| uv selects an old/broken system Python | Run `uv python install 3.12`, then `uv sync --locked --managed-python` |
| `uvicorn` or a package cannot be found | Use `uv run --locked python -m uvicorn ...`; with pip, activate the correct venv |
| `.env` or sample data is not found | Start commands from the repository root; restart after changing settings |
| Health works but analysis returns 503 | Configure the model credentials/endpoint; health does not check model readiness |
| Codex backend fails | Run `codex login status`, confirm `CODEX_MODEL` access, and check the local CLI version and per-call timeout |
| Analysis returns 200 with errors or missing sections | Inspect `errors`; check model connectivity, JSON output, model name, and server logs |
| Local model connection fails | Start that server separately; use its actual model ID and endpoint, often ending in `/v1` |
| Docker cannot reach the host model | Use `host.docker.internal`, not `localhost`; the model must accept host connections |
| Port 8000 is occupied | Change the API `--port` or Compose host port and use that port in client requests |
| A model call times out | Check backend load and client timeout; `LLM_TIMEOUT_SECONDS` applies per model call |
| Evaluation output cannot be written | Choose a new writable `--output-dir`; do not write into the historical archive |

## Scope and attribution

The audit stage runs after model calls and does not redact returned patient data. Remote model configuration sends narratives and intermediate data to the selected endpoint. Authentication and durable audit storage are not implemented.

Bundled medical mappings and rules are incomplete. FHIR helpers are experimental and outside the pipeline. Historical evidence does not validate clinical accuracy, safety, throughput, FHIR conformance, or HIPAA compliance.

Adapted from [bcefghj/medical-multi-agent-system](https://github.com/bcefghj/medical-multi-agent-system) under [MIT](LICENSE). See [attribution and project history](ATTRIBUTION.md) for provenance.

## Current limitations

The recorded runs establish that the demo can execute and archive its outputs. The following gaps remain in the current implementation; no database or model-quality upgrade below has been implemented or benchmarked.

| Area | Observed limitation | Consequence |
|---|---|---|
| Coding labels and validation | A single unreviewed fixture code is the target; labels mix specificity levels and have no coding-year/encounter contract. Coding uses LLM JSON directly and never invokes the bundled catalog validator. | Exact label agreement is not coding correctness. Category-prefix agreement can also accept an invalid suffix. Generated DRG values are not produced by an authoritative grouper. |
| Information between Agents | Coding receives Diagnosis and Treatment, but not `patient_info` or the original narrative. Intake can estimate/default missing values; later stages mainly parse dictionaries rather than validate the full domain schemas. | Patient facts can be lost or inferred without an adequate source trail. In the saved runs, hypertension and smoking-cessation details present in Intake do not reach Coding. |
| Knowledge retrieval | Ten exact symptom keys and 15 disease-to-code hints form the bundled lookup. Synonyms, negation, context, citations, graph traversal, and embeddings are not modeled. | Candidate recall is brittle. A name such as `GraphRAG` does not demonstrate evidence-backed reasoning or an improvement over a simple baseline. |
| Missing information and retries | A loopback reruns Intake on the same narrative. At the configured cap, Treatment continues even if more information was requested; the matched comparison deliberately uses zero loopbacks. | The graph can terminate without resolving uncertainty. More complete-looking output is not proof of sufficient clinical information. |
| Evaluation coverage | Two synthetic cases per current model, one repeat, substring diagnosis aliases without negation handling, and structural completion checks. No held-out cohort, clinician/coder review, single-model baseline, or retrieval ablation. | Tied scores and different timings on two observations do not establish comparative clinical quality, reliability, or GraphRAG benefit. |
| Medication and audit checks | Curated DDI/allergy rules cover a small subset; medication presence and execution stamps do not assess dose or treatment appropriateness. Audit runs after model calls. | Returning a plan/audit is not a safety or compliance result. The current pipeline does not redact all data before remote inference. |
| Persistence and serving | No default patient database, durable audit store, resumable API conversation, authentication, or distributed job queue. | JSON experiment archives do not provide a multi-user application storage or access-control layer. |
| Latency and failure evidence | Four sequential model stages, CLI startup/transport overhead, and possible loopbacks. The earlier `ultra` pilot timed out; timeout traces lack a completed event stream. | Latency depends on the full protocol and backend. Empty timeout usage/tool fields are missing evidence; token/cost totals cannot be treated as complete. |

### Storage and retrieval alternatives

There is no active Neo4j deployment to replace today. The choices below would add capabilities to the current in-memory prototype. Patient/session persistence and knowledge retrieval are separate concerns, even when one database can support both.

| Proposed option | Good fit | What it would add / require |
|---|---|---|
| SQLite + [FTS5](https://www.sqlite.org/fts5.html) | A local, single-user demo with durable structured records and lexical document search | A small local database and full-text index; no embedding provider required for lexical search. Requires a schema, ingestion, retrieval adapter, and checking FTS5 availability in the Python SQLite build. |
| PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) | A deployed service needing relational records, metadata filters, and vector retrieval together | A PostgreSQL service, vector extension, document/embedding ingestion, and an adapter. pgvector supports exact/approximate neighbor search; embeddings and retrieval quality must be evaluated separately. |
| [Qdrant local mode](https://qdrant.tech/documentation/frameworks/langchain/#local-mode), then a server if needed | A Python vector-search prototype or a dedicated retrieval service | In-memory or on-disk local vector storage can avoid running a server during development. Requires a client, an embedding provider, ingestion, and an adapter; application records still need a storage design. |
| Neo4j with an ingested graph | Explicit disease/symptom/drug relationships and graph queries that the task actually needs | A graph schema, source/versioned ingestion, queries integrated into Diagnosis, and evaluation. Neo4j can combine graph and vector capabilities, but the existing connection helper provides neither automatically. |

For this repository's small local demo, **SQLite + FTS5 is the first extension to evaluate**. For semantic retrieval experiments, Qdrant local mode is another lightweight development option; for a multi-user service, evaluate PostgreSQL + pgvector. Keep Neo4j for a demonstrated graph-relation requirement. This is an architectural recommendation, not an observed ranking of retrieval or model quality.

A different database alone does not fix incomplete knowledge, incorrect labels, lost patient facts, or prompt bias. Compare retrieval methods on the same reviewed corpus and held-out cases, with fixed model, top-k/context and token budgets. Measure evidence recall, unsupported outputs, coding validity/acceptability, failures, and latency before claiming a gain.

**Compatibility requires an adapter.** The current synchronous retrieval interface and prompt wording assume `symptom_match_count`. A vector similarity score cannot be substituted for that integer. Preserve existing result/trace fields, make scores and prompt descriptions method-aware, and explicitly integrate synchronous or asynchronous clients. Merely changing `.env` is insufficient. See the [proposed compatibility contract](docs/architecture.md#retrieval-backend-compatibility). The existing Codex adapter supplies text responses; an embedding interface is not implemented.

## Upgrade and extension roadmap

These are proposed steps, ordered by the evidence and engineering gaps above. Existing archived scores and raw outputs stay unchanged; new evaluations should have a new dataset/protocol version.

| Priority | Upgrade / extension | Acceptance evidence |
|---|---|---|
| 1. Repair the evaluation contract | Pin ICD-10-CM release, encounter assumptions and label granularity; independently review acceptable codes. Add catalog membership, complete-code specificity, and adjudicated acceptability measures alongside the existing exact match. | A reviewed, versioned label set; separate error analysis for invalid codes, specificity differences, and clinically unacceptable assignments. Do not derive the answer key from current model predictions. |
| 2. Preserve and validate clinical facts | Pass sourced patient facts to Coding, validate every stage with typed schemas, distinguish known/unknown/inferred fields, and route unresolved information to an explicit clarification/review state. | Handoff tests that preserve relevant facts, invalid-output tests, and termination/clarification cases that do not invent new observations. |
| 3. Add local persistence and a retrieval boundary | Start with a SQLite schema for experiment/session metadata and a separately versioned knowledge corpus; retain JSON export. Introduce interchangeable lexical/vector/graph adapters and method-aware evidence traces. | Persistence across restarts, trace/export parity, adapter contract tests, and a default offline path with no external database requirement. |
| 4. Evaluate richer retrieval and grounded coding | Normalize symptom terminology; ingest licensed/versioned sources with citations. Test lexical, dense, hybrid and graph retrieval; constrain/rank Coding candidates using a complete code catalog. Use a real grouper if DRG claims are required. | Held-out retrieval and end-to-end ablations under the same input/model/budget; catalog-valid output and reviewed coding/safety checks. More infrastructure is not an acceptance criterion. |
| 5. Expand the benchmark | Add diverse reviewed cases, ambiguous and adversarial narratives, more repeats, model/backend settings, missing-information cases, and a simple single-model baseline. Measure latency percentiles, error recovery, and complete usage where available. | Versioned raw traces and failure-inclusive reports with appropriate denominators and uncertainty; no post-hoc label changes to improve a score. |
| 6. Harden a deployed service | Add authentication/access boundaries, durable audit storage, pre-transmission data handling, retention controls, background jobs, timeouts/cancellation, observability, and tested FHIR integration. | Integration and failure-recovery tests, access/storage checks, and independently reviewed deployment controls before using sensitive data or making compliance claims. |
