# Architecture

The application is a Python package under `src/medical_multi_agent`. FastAPI accepts requests; a LangGraph `StateGraph` executes five stages over `ClinicalState`. The implementation is a sequential workflow with bounded diagnostic retries.

[![Current system architecture](assets/architecture/system-architecture.png)](assets/architecture/system-architecture.svg)

[![Five-Agent collaboration and bounded retry](assets/architecture/agent-workflow.png)](assets/architecture/agent-workflow.svg)

## Execution and state

| Stage | Reads | Produces |
|---|---|---|
| Intake | Narrative | Structured patient information |
| Diagnosis | Patient information and bundled retrieval candidates | Primary and differential diagnoses, retrieval trace, retry request |
| Treatment | Patient information and diagnosis | LLM treatment proposal plus deterministic DDI/allergy checks |
| Coding | Diagnosis and treatment | LLM coding result using illustrative ICD-10/DRG guidance |
| Audit | Prior outputs | PHI-pattern findings, audit records, unverified deployment controls |

Intake, diagnosis, treatment, and coding use the shared `config/llm.py` factory. Audit uses deterministic rules. JSON generation can fail; callers must inspect `errors` and missing output sections.

The optional `LLM_BACKEND=codex` uses a separately authenticated Codex CLI instead of the application's HTTP model client. Each invocation starts a fresh ephemeral call in a temporary directory, passing the original system prompt through an instructions file and the human message through stdin. The CLI sends these inputs to a remote model service; a temporary working directory does not make inference local or air-gapped. The four model stages and deterministic Audit retain the same workflow.

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

## Retrieval backend compatibility

The current active backend is the in-memory map. No SQLite, PostgreSQL, Qdrant, vector embedding, or Neo4j retrieval adapter is implemented. A database connection alone does not change the retrieval algorithm or supply a medical corpus.

### Current call and result contract

`Diagnosis` calls the synchronous `find_diseases_by_symptoms(symptoms: list[str]) -> list[dict]`. Each map result has `disease`, an integer `symptom_match_count`, `icd10_code`, and `icd10_description`. Diagnosis takes up to five candidates, formats overlap-count hints, and records the candidates in `diagnosis.knowledge_graph_candidates` with a `knowledge_sources` entry.

The evaluation reads candidate count and the leading `disease`; the prompt formatter also reads `symptom_match_count`. Thus a vector client cannot replace the backend by returning a cosine score in the count field, or raw document hits without a disease mapping. Neo4j queries likewise need an explicit graph-to-candidate mapping.

### Proposed adapter boundary

This is a design for future implementation, not a ready-to-use configuration option:

| Boundary | Required behavior |
|---|---|
| Selection and lifecycle | Add an explicit retrieval factory/backend setting, initialize and close clients deliberately, and keep the bundled offline backend as the default. |
| Candidate identity | Retain disease and code/description fields and the existing API trace key for compatibility; map document/graph evidence to candidates explicitly. |
| Scores and evidence | Keep real lexical overlap counts separate from similarity/distance and reranker scores. Include method, score type, source IDs, source/catalog versions, and evidence excerpts. Use unknown/absent counts where no overlap count is measured; update the formatter and schema before enabling that result type. |
| Model context | Make prompt wording method-aware: lexical counts, semantic matches, and graph paths are different evidence. Preserve provenance and avoid asserting a citation supports a model claim merely because a hit was retrieved. |
| Execution model | A synchronous client fits the current node boundary. An async Neo4j/Qdrant client requires awaiting in the retrieval/node path and compatible graph/API invocation; returning a coroutine through the present synchronous method will not work. |
| Embeddings | Add a separately configured and versioned embedding provider, dimensions, chunking/ingestion and index metadata. The current text-generation factory/Codex adapter supplies no embedding interface. |
| Failure and migration | Define timeout/unavailable-store behavior, explicit fallbacks, schema migrations and export parity. Record fallback use instead of silently treating it as the configured backend. |

Patient/session persistence, retrieval knowledge, checkpoints, and evaluation archives are separate storage responsibilities. A single SQL service may host several of them, but swapping the retriever does not automatically add durable API sessions, audit retention, or FHIR integration.

Before enabling a new backend, test result schema/trace compatibility, corpus and source-version reproducibility, failures/fallbacks, and retrieval relevance under a fixed model/context budget. The [README roadmap](../README.md#upgrade-and-extension-roadmap) prioritizes label and handoff repair before claiming backend quality improvements.

## Privacy and audit boundaries

The audit node scans selected structured outputs using a limited regex set. It may flag ordinary clinical phrases or miss identifiers. It does not scan and redact all data before LLM transmission, and it does not replace pipeline outputs with redacted versions.

Deployment properties such as encryption, access controls, retention, and breach handling cannot be inferred from a narrative. The audit report marks them unverified. Legacy response keys such as `hipaa_compliant` remain for compatibility and should not be treated as compliance certification.

A local LLM endpoint can keep model inference on infrastructure you control, depending on how that endpoint is configured. Choosing a remote endpoint sends inputs there. The repository does not enforce network isolation, implement authentication, provide WORM storage, or establish HIPAA compliance.

## Configuration

Settings load from environment variables and `.env` in the directory where the command starts. The LLM backend, timeout, reasoning-template option, and diagnostic retry budget are described in `.env.example`. Settings are cached per process; restart after changing configuration.

The default runtime requires no PostgreSQL, Redis, Neo4j, or FHIR server. Docker Compose therefore starts only the API. `/health` reports process liveness and does not exercise the LLM or tools.

## Diagram sources

Regenerate both diagrams without model or network calls after installing the optional plotting dependencies:

```bash
uv run --locked --extra plots python -m evaluation.draw_architecture
```

The generator writes HTML previews, accessible SVGs, and PNGs to `docs/assets/architecture/`. Use `--output-dir` for another destination. The diagrams follow the evaluation figures' white background, blue accent, and offline DejaVu fonts. The HTML previews contain inline SVG and require no external assets.

The architecture view summarizes components and calls. The workflow view follows the numbered execution order, separates the four model stages from deterministic Audit, and records the precise retry condition. It also distinguishes LLM coding proposals from the standalone deterministic ICD-10/DRG lookup endpoints.
