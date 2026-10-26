# Architecture

The application is a Python package under `src/medical_multi_agent`. FastAPI accepts requests; a LangGraph `StateGraph` executes five stages over `ClinicalState`. The implementation is a sequential workflow with bounded diagnostic retries.

## Execution and state

| Stage | Reads | Produces |
|---|---|---|
| Intake | Narrative | Structured patient information |
| Diagnosis | Patient information and bundled retrieval candidates | Primary and differential diagnoses, retrieval trace, retry request |
| Treatment | Patient information and diagnosis | LLM treatment proposal plus deterministic DDI/allergy checks |
| Coding | Patient information, diagnosis, treatment | LLM coding result using illustrative ICD-10/DRG guidance |
| Audit | Prior outputs | PHI-pattern findings, audit records, unverified deployment controls |

Intake, diagnosis, treatment, and coding use the shared `config/llm.py` factory. Audit uses deterministic rules. JSON generation can fail; callers must inspect `errors` and missing output sections.

The API executes the synchronous graph in FastAPI's worker thread pool. Each request receives an independent internal execution identifier, including requests that reuse a supplied `thread_id` label. The label does not implement a resumable clinical conversation.

The default graph keeps state only for the invocation and does not retain checkpoint history. Library callers can explicitly pass a LangGraph checkpointer to `build_clinical_pipeline(checkpointer=...)`; its storage and retention then depend on that caller-provided implementation. Audit records are returned within the result. There is no implemented database persistence, cache, or distributed worker queue.

## Diagnostic retry behavior

`MAX_DIAGNOSIS_RETRIES` controls diagnosis-to-intake loopbacks, with a default of two. The initial diagnosis attempt is additional to that retry budget. A retry reprocesses the original narrative; it does not obtain missing tests or ask a human a follow-up question.

When the budget is exhausted, the graph proceeds to treatment with the available result. The budget bounds graph loops, while model timeouts and provider retries determine how long a request may take. This behavior is a demonstration of routing rather than a validated clinical escalation policy.

## Retrieval and deterministic tools

`services/graphrag_service.py` ranks diseases by exact normalized symptom-key overlap in bundled dictionaries. Diagnosis includes up to five candidates in its prompt and stores them in `knowledge_graph_candidates`. Unrecognized symptom wording may retrieve nothing. The retrieval trace shows what the model received; it does not prove that the model used it correctly.

Neo4j connection and Cypher helpers remain available with the optional `neo4j` dependency:

```bash
python -m pip install -e '.[neo4j]'
```

Installing this extra does not switch diagnosis to a graph database. There is no bundled graph ingestion pipeline, UMLS/SNOMED corpus, vector index, or multi-hop retrieval evaluation.

Treatment runs curated interaction and allergy checks after the LLM proposes medications. These checks can add warnings and execution stamps. They do not establish medication appropriateness, validate doses, or comprehensively cover interactions.

ICD-10 lookup and DRG grouping use small illustrative mappings. They are not a complete current ICD-10-CM catalog or an implementation of CMS grouping logic.

FHIR helpers convert dictionaries and can submit resources to a configured server if explicitly called. They are outside the default workflow. Current outputs have known limitations, including estimated birth dates and unvalidated medication dose values. No hospital integration or FHIR conformance result is claimed.

## Privacy and audit boundaries

The audit node scans selected structured outputs using a limited regex set. It may flag ordinary clinical phrases or miss identifiers. It does not scan and redact all data before LLM transmission, and it does not replace pipeline outputs with redacted versions.

Deployment properties such as encryption, access controls, retention, and breach handling cannot be inferred from a narrative. The audit report marks them unverified. Legacy response keys such as `hipaa_compliant` remain for compatibility and should not be treated as compliance certification.

A local LLM endpoint can keep model inference on infrastructure you control, depending on how that endpoint is configured. Choosing a remote endpoint sends inputs there. The repository does not enforce network isolation, implement authentication, provide WORM storage, or establish HIPAA compliance.

## Configuration

Settings load from environment variables and `.env` in the directory where the command starts. The LLM backend, timeout, reasoning-template option, and diagnostic retry budget are described in `.env.example`. Settings are cached per process; restart after changing configuration.

The default runtime requires no PostgreSQL, Redis, Neo4j, or FHIR server. Docker Compose therefore starts only the API. `/health` reports process liveness and does not exercise the LLM or tools.
