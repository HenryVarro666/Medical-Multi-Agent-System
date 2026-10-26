# Attribution and provenance

This repository began with the MIT-licensed scaffold [bcefghj/medical-multi-agent-system](https://github.com/bcefghj/medical-multi-agent-system). The original import is recorded in commit `75ce75e` (`Import open-source scaffold: bcefghj/medical-multi-agent-system (MIT)`). The inherited MIT notice is preserved in [LICENSE](LICENSE).

The scaffold supplied the five-agent workflow, API and model structure, bundled medical lookup tables, synthetic case fixtures, service helpers, and initial documentation. These are adapted upstream materials, not independently developed clinical infrastructure.

Commit `9805eb4` recorded this fork's symptom-retrieval integration, deterministic DDI reconciliation, diagnostic retry guard, configurable local LLM endpoint, and evaluation harness. Its recorded July 13, 2026 evaluation is preserved under `evaluation/results/historical-2026-07-13/`.

The Python-only refactor consolidates the package, removes alternate-language implementations and interview materials, replaces unsupported feature claims, and adds packaging, offline tests, and safer evaluation output handling. See Git history and the current diff for the precise ownership and scope of changes.

## Data and evidence

`data/sample_patients.json` contains five synthetic vignettes inherited from the scaffold. The repository has no documented clinician adjudication, independent labeling study, or real patient dataset. Expected labels are illustrative and may omit clinically relevant specificity.

Symptom mappings, ICD-10 entries, DRG mappings, drug interactions, and allergy rules are small demonstration tables inherited or adapted from the scaffold. They have not been validated as a complete or current clinical reference. References to ICD-10, FHIR, or HIPAA describe the intended domain; they do not establish endorsement, certification, official data licensing, or conformance.

Historical raw model outputs and summaries are retained unchanged. Their original prose includes claims and field names that should be interpreted using the caveats in [evaluation/README.md](evaluation/README.md). Those artifacts describe the older code and run environment, not the results of the present refactor.

## Third-party software

The project uses LangGraph, LangChain integrations, FastAPI, Pydantic, HTTPX, structlog, Uvicorn, and optionally the Neo4j driver. Their respective licenses apply to those dependencies. A model backend, its weights, and its serving stack are separate dependencies governed by their own terms.


## Packaged ICD-10-CM reference

Coding-validation v2 uses an offline catalog derived from the [CMS FY2027 Code Descriptions in Tabular Order archive](https://www.cms.gov/files/zip/2027-code-descriptions-tabular-order.zip). The packaged compressed data and manifest preserve the release, source/archive/member hashes, official descriptions, and complete-code/header distinctions. `evaluation.build_icd10_catalog` rebuilds those bytes from a separately downloaded, hash-verified source archive without network access. This is external reference data, not a code set authored by this project; applicable source rights and terms remain with their owners.

The v2 research fixture refines the original case_002 parent label using the [official FY2027 Index/Tabular](https://www.cms.gov/files/zip/2027-code-tables-tabular-index.zip). Narratives and the original fixture file remain unchanged; the change and its limits are versioned. Catalog review is not independent clinical coding adjudication, endorsement, or a validated DRG grouping implementation.
