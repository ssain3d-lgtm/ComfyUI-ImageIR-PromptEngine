# llama.cpp Model Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a ComfyUI llama.cpp router model manager that can connect/refresh, select, load and unload models, then emit the existing `IMAGEIR_BACKEND` config.

**Architecture:** Keep router HTTP logic in a standard-library-only backend module, keep ComfyUI proxy/UX code thin, and leave the existing single-model launcher untouched. A small frontend extension updates the model dropdown and buttons while the Python node remains queue-executable without JS.

**Tech Stack:** Python 3.10+, urllib/json/dataclasses, ComfyUI node API + PromptServer routes, vanilla JavaScript frontend extension.

**Spec:** `docs/superpowers/specs/2026-09-15-llama-model-manager-design.md`

## Global Constraints

- No new runtime Python dependencies.
- Existing `Image IR Backend Config` and `Image IR Backend Control` remain compatible.
- No token value may be returned to frontend/debug output.
- Core `imageir` package must import without ComfyUI installed.
- Router manager targets llama.cpp `GET /models`, `GET /models?reload=1`, `POST /models/load`, and `POST /models/unload`.

---

### Task 1: Router client

**Files:**
- Create: `imageir/backend/llama_router.py`
- Test: `tests/test_llama_router.py`

**Interfaces:**
- Produces `LlamaRouterClient`, `RouterModel`, `RouterError`.
- `list_models(reload=False) -> tuple[RouterModel, ...]`
- `load_model(model_id: str) -> None`
- `unload_model(model_id: str) -> None`

- [ ] Write failing tests for list parsing, `reload=1`, load/unload payloads, vision metadata, malformed replies and masked HTTP errors.
- [ ] Run the focused test module and verify RED.
- [ ] Implement minimal stdlib HTTP client and typed parser.
- [ ] Run focused tests and verify GREEN.

### Task 2: ComfyUI manager node

**Files:**
- Modify: `image_ir_backend_nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Add `ImageIRLlamaModelManager` display name `ImageIR Llama.cpp Model Manager`.
- Inputs: `base_url`, `selected_model`, `action`, generation settings, optional `api_token_env`/direct token.
- Outputs: `IMAGEIR_BACKEND`, `STRING models_json`, `STRING status`, `STRING selected_model`.

- [ ] Write failing tests that the node emits a connect-existing llama config with the selected model and that action routing uses the router client without exposing secrets.
- [ ] Run focused node tests and verify RED.
- [ ] Implement the node with dependency injection seam for the router client.
- [ ] Run focused tests and verify GREEN.

### Task 3: Frontend proxy and dynamic dropdown

**Files:**
- Create: `image_ir_web.py`
- Create: `web/llama_model_manager.js`
- Modify: `__init__.py`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Proxy endpoint accepts only normalized router actions and server URL/model id; token values remain server-side only when coming from queued node execution.
- `WEB_DIRECTORY = "./web"`.
- JS enhances only `ImageIRLlamaModelManager`, updates combo options, and exposes Connect/Refresh/Load/Unload buttons.

- [ ] Write failing packaging/import tests for `WEB_DIRECTORY`, JS asset presence, and bare import without ComfyUI.
- [ ] Verify RED.
- [ ] Implement optional route registration guarded by ComfyUI availability plus frontend extension.
- [ ] Verify focused tests GREEN.

### Task 4: Docs and regression verification

**Files:**
- Modify: `README.md`
- Test: existing full suite.

- [ ] Document router-mode startup (`--models-dir`), manager workflow, old-server fallback and single-model compatibility.
- [ ] Run Ruff + all tests + coverage in CI.
- [ ] Inspect CI failures and fix only regressions related to this feature.
- [ ] Open a PR from `feature/llama-model-manager` to `main` after CI is green.
