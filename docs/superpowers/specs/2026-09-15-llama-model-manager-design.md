# llama.cpp Model Manager Design

## Goal

Add a convenience layer on top of the existing `IMAGEIR_BACKEND` path so a user can connect to a llama.cpp router server, refresh the available model list, select one from a dropdown, load/unload it, and emit a normal `BackendConfig` for the existing analyzer/intent nodes.

## Scope

This feature is additive. It does not replace `Image IR Backend Config` or `Image IR Backend Control` and must not break single-model llama-server usage, LM Studio, Gemini, or existing workflows.

## llama.cpp router contract

The manager targets current llama.cpp router mode started with `llama-server --models-dir <dir>`. It uses:

- `GET /models` to list models and status.
- `GET /models?reload=1` to refresh the model source.
- `POST /models/load` with `{ "model": "<id>" }`.
- `POST /models/unload` with `{ "model": "<id>" }`.

The model list includes status and architecture metadata. Vision-capable models are detected from `architecture.input_modalities` containing `image`.

## Architecture

1. `imageir/backend/llama_router.py` contains a standard-library-only client. It owns HTTP request/response parsing and returns typed model metadata. It has no ComfyUI dependency.
2. `ImageIR Llama.cpp Model Manager` is a ComfyUI node that consumes server settings, selected model, and action (`status`, `refresh`, `load`, `unload`). It returns a normal `IMAGEIR_BACKEND`, a JSON model list, a readable status, and selected model id.
3. `web/llama_model_manager.js` enhances that node only. It adds Connect/Refresh/Load/Unload buttons and keeps the `selected_model` combo values synchronized with the Python proxy endpoint.
4. A thin ComfyUI-only route module proxies router API operations so the browser never talks directly to arbitrary localhost ports and is not blocked by CORS. Importing the core package without ComfyUI remains valid.
5. The existing backend config and launcher stay available for advanced/manual setup.

## UX

The manager node shows:

- server URL
- selected model combo
- Connect / Refresh
- Load / Unload
- status text

On Connect/Refresh it fetches `/models`, updates the dropdown, and marks each entry by state and vision capability in the UI label while preserving the real llama.cpp model id internally. Choosing a model updates the backend model id. Load and Unload call the router endpoints and then refresh state.

The Python node path remains usable without JavaScript: setting the combo/model id and queueing the node with an action performs the same operation.

## Safety and compatibility

- Never stop the llama-server process merely to change models.
- Never shell out from the browser.
- Reuse the existing masked token handling; direct tokens are never returned in frontend responses.
- Treat HTTP failures, unsupported old llama.cpp servers, malformed JSON, missing model ids, and router-mode absence as readable errors.
- Do not auto-load a model on simple status/refresh.
- A selected model produces a `BackendConfig(provider="llama_cpp", server_mode="connect_existing", model_name=<id>)` so all existing OpenAI-compatible generation code remains unchanged.

## Testing

Add unit tests for router list parsing, refresh URL, load/unload payloads, vision capability detection, token masking, node backend generation, and unsupported-server errors. Extend packaging tests to cover the new web directory without adding third-party runtime dependencies.
