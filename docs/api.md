# API reference

Start the API as described in the root README. Interactive schemas are available at `/docs`; the OpenAPI document is at `/openapi.json`.

| Method | Path | Backend required |
|---|---|---|
| GET | `/health` | None |
| POST | `/api/v1/clinical/analyze` | Configured LLM |
| POST | `/api/v1/clinical/icd10/search` | None |
| GET | `/api/v1/clinical/icd10/{code}` | None |
| GET | `/api/v1/clinical/icd10/validate/{code}` | None |
| POST | `/api/v1/clinical/ddi/check` | None |

## Full analysis

```http
POST /api/v1/clinical/analyze
Content-Type: application/json
```

```json
{
  "patient_description": "A synthetic narrative of at least ten characters.",
  "thread_id": "optional-request-label"
}
```

`thread_id` is optional. Reusing it does not resume prior patient state; each analysis has an independent internal execution identifier.

A successful HTTP response has these fields:

```json
{
  "patient_info": {},
  "diagnosis": {},
  "treatment_plan": {},
  "coding_result": {},
  "audit_result": {},
  "errors": []
}
```

This example shows the response envelope only. Sections contain model-dependent values and may be `null` after a failed stage. Inspect `errors` before using outputs. Retrieval candidates appear under `diagnosis.knowledge_graph_candidates`; deterministic check stamps appear under `treatment_plan.ddi_engine`.

The audit output contains regex findings and deployment controls marked unverified. It does not redact the other response sections. A response is not a clinical recommendation or a HIPAA compliance assessment.

Request validation errors return HTTP 422. Missing remote-backend credentials return HTTP 503. Unhandled pipeline errors return HTTP 500; agent-level failures may instead produce HTTP 200 with partial results and a nonempty `errors` list.

## ICD-10 search

```bash
curl -s http://127.0.0.1:8000/api/v1/clinical/icd10/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"pneumonia"}'
```

The response contains `query`, `results`, and `count`. The query must be at least two characters. Results come from the bundled demonstration catalog.

## ICD-10 lookup

```bash
curl -s http://127.0.0.1:8000/api/v1/clinical/icd10/J18.9
```

The response contains `icd10` and `drg_group`. Unknown codes return HTTP 404. Absence from this small catalog does not mean a code is invalid in an official code set. DRG results use simplified mappings.

## Official ICD-10-CM validation

```bash
curl --fail http://127.0.0.1:8000/api/v1/clinical/icd10/validate/I21.09
```

This offline endpoint checks the packaged CMS FY2027 catalog. It returns the normalized code, `valid`, `reason`, `is_complete_code`, official description and catalog release. `I21.09` is complete; `I21.0` returns `valid: false` with `reason: non_reportable_header`. Malformed and unknown codes also return an explicit invalid verdict. These verdicts do not adjudicate whether a code is supported by a particular patient record.

The older search/lookup endpoints above still use the small demonstration tables, so a lookup 404 is not an authoritative validation verdict.

Coding now receives the original narrative and full patient record and adds `code_validation`, `requires_review`, `clinical_review_required`, `model_proposal` and `raw_model_output` to its result. Primary and every secondary code are checked; invalid codes or code-bearing JSON shapes set `requires_review` and add a pipeline error. Raw proposals remain visible. Valid descriptions are replaced with catalog descriptions while original descriptions are retained. `clinical_review_required` remains true even when catalog checks pass. DRG proposals remain unverified and may be null.

## Drug interaction check

```bash
curl -s http://127.0.0.1:8000/api/v1/clinical/ddi/check \
  -H 'Content-Type: application/json' \
  -d '{"new_drugs":["warfarin"],"current_drugs":["aspirin"]}'
```

`new_drugs` must contain at least one string; `current_drugs` defaults to an empty list. The response includes `interactions`, `interaction_count`, and `has_major_interaction`. The standalone endpoint checks the curated DDI rules; allergy checks occur separately in treatment. No detected interaction means no matching bundled rule, and does not establish safety.

## Health

```bash
curl -s http://127.0.0.1:8000/health
```

This endpoint reports API liveness. It does not test model connectivity, authentication, clinical validity, or external service readiness.

The API has no authentication and should be used locally with synthetic fixtures. Its model-dependent analysis can take much longer than deterministic lookups; configure client timeouts accordingly.
