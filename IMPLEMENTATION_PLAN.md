# FocusAI – Implementation Plan and Status

Legend: **DONE+VERIFIED** (executed here, result observed) · **WRITTEN, NOT EXECUTED** (code/tests exist, sandbox could not run them) · **TODO**

| # | Requirement | Status |
|---|-------------|--------|
| R1 | Genuine Integrated Gradients on all-MiniLM-L6-v2, target = cosine(sentence, study context) | IG kernel DONE+VERIFIED (numpy, analytic functions). Torch/MiniLM path WRITTEN, NOT EXECUTED |
| R2 | Existing pipeline, thresholds, schemas, contracts unchanged | DONE+VERIFIED by diff: relevance_engine, text_utils, context, llm, content_extractor byte-identical; `explainability.py` docstring only; `/analyze` untouched |
| R3 | Word-level heatmap in extension, sentence heatmap kept | DONE+VERIFIED in real Chromium (stubbed backend) |
| R4 | Endpoint + cache + validation + graceful failure | Schema validation DONE+VERIFIED; endpoint/service WRITTEN, NOT EXECUTED (FastAPI/SQLAlchemy unavailable) |
| R5 | Tests | See PROJECT_PROGRESS.md matrix |
| R6 | Audit of rest of project | Static audit DONE; no features executed |
| R7 | Docs | DONE (backend/README.md, README.md, this file, PROJECT_PROGRESS.md) |

## Phases
1. Audit – DONE (static; baseline backend tests could not run, see blocker).
2. IG module – DONE (`backend/app/ai/integrated_gradients.py`); torch path unexecuted.
3. Backend integration – DONE (`POST /api/relevance/explain-tokens`, LRU cache); unexecuted.
4. Extension integration – DONE+VERIFIED (background handler + shadow-DOM panel).
5. Regression – partial (see matrix).
6. Completion audit – static only; BOM removed from `.env.example`, stale test mock fixed.

## Design (as built)
* Gradients w.r.t. word embeddings (`inputs_embeds`) → BERT → masked mean pool → L2 normalise → dot with fixed context vector.
* Baseline: [PAD] embedding at non-special positions; [CLS]/[SEP] kept (zero attribution).
* Trapezoid rule, 32 steps (max 64), auto-doubling once if the completeness check fails; result reports `converged`.
* Words = regex words/punctuation; sub-word attributions **summed**; raw signed values + display-only normalised values.
* Lexical fallback / unsupported model → HTTP 200 `available:false` + reason. Core analysis never depends on it.
* Scope statement is returned by the API and shown in the UI: only the cosine component is explained.

## Files changed vs. the original ZIP
New: `backend/app/ai/integrated_gradients.py`, `backend/tests/test_ig_core.py`, `backend/tests/test_integrated_gradients.py`,
`extension-tests/browser/ig-ui.browser.js`, `IMPLEMENTATION_PLAN.md`, `PROJECT_PROGRESS.md`.
Edited: `backend/app/ai/embeddings.py` (+`st_model` read-only property), `backend/app/ai/explainability.py` (docstring only),
`backend/app/api/relevance.py`, `backend/app/schemas/relevance.py`, `backend/app/services/relevance_service.py` (additions only),
`extension/background.js`, `extension/content.js` (additions + reset on navigation), `extension-tests/tests/background.test.js`
(mock fix + 6 tests), `backend/.env.example` (BOM removed), `backend/README.md`, `README.md`.
Not changed: manifest, DB schema/models, auth, tracking, thresholds, scoring, `/analyze` contract, frontend dashboard.
