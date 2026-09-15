import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "ImageIRLlamaCppModelManager";
const ENDPOINT = "/imageir/llama/models";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function value(node, name, fallback = "") {
    const widget = findWidget(node, name);
    return widget ? widget.value : fallback;
}

function payload(node, action, selected) {
    return {
        action,
        model_id: selected || value(node, "model_id", ""),
        connection_mode: value(node, "connection_mode", "connect_existing"),
        base_url: value(node, "base_url", "http://127.0.0.1:8080"),
        api_token_env: value(node, "api_token_env", ""),
        llama_server_path: value(node, "llama_server_path", "llama-server"),
        models_dir: value(node, "models_dir", ""),
        host: value(node, "host", "127.0.0.1"),
        port: Number(value(node, "port", 8080)),
        models_max: Number(value(node, "models_max", 1)),
        models_autoload: Boolean(value(node, "models_autoload", false)),
        router_extra_args: value(node, "router_extra_args", ""),
        context_size: Number(value(node, "context_size", 32768)),
        gpu_layers: Number(value(node, "gpu_layers", 999)),
        model_extra_args: value(node, "model_extra_args", "--jinja"),
        max_tokens: Number(value(node, "max_tokens", 4096)),
        temperature: Number(value(node, "temperature", 0.2)),
        top_p: Number(value(node, "top_p", 0.9)),
        timeout: Number(value(node, "timeout", 120.0)),
    };
}

async function callManager(node, action, selector, statusWidget) {
    const selected = selector.value || value(node, "model_id", "");
    statusWidget.value = `${action}...`;
    node.setDirtyCanvas(true, true);
    try {
        const response = await api.fetchApi(ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload(node, action, selected)),
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        const models = Array.isArray(data.models) ? data.models : [];
        const ids = models.map((model) => model.id);
        selector.options.values = ids.length ? ids : [""];

        let next = value(node, "model_id", "");
        if (!ids.includes(next)) {
            next = models.find((model) => model.status === "loaded" && model.vision)?.id
                || models.find((model) => model.status === "loaded")?.id
                || ids[0]
                || "";
        }
        selector.value = next;
        const backing = findWidget(node, "model_id");
        if (backing) backing.value = next;

        const selectedInfo = models.find((model) => model.id === next);
        const selectedStatus = selectedInfo
            ? `${selectedInfo.status}${selectedInfo.vision ? " · vision" : " · text"}`
            : "no model selected";
        statusWidget.value = `${data.status || "connected"}\n${models.length} model(s) · ${selectedStatus}`;
    } catch (error) {
        statusWidget.value = `ERROR: ${error?.message || error}`;
    }
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ImageIR.LlamaCppModelManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const backing = findWidget(this, "model_id");

            const selector = this.addWidget(
                "combo",
                "available_models",
                backing?.value || "",
                (selected) => {
                    if (backing) backing.value = selected || "";
                },
                { values: backing?.value ? [backing.value] : [""] },
            );
            selector.serialize = false;

            const statusWidget = this.addWidget(
                "text",
                "router_status",
                "Not connected — press Connect / Refresh",
                null,
                { multiline: true },
            );
            statusWidget.serialize = false;

            this.addWidget("button", "Connect / Refresh", null, () =>
                callManager(this, "connect", selector, statusWidget));
            this.addWidget("button", "Load", null, () =>
                callManager(this, "load", selector, statusWidget));
            this.addWidget("button", "Unload", null, () =>
                callManager(this, "unload", selector, statusWidget));

            const size = this.computeSize();
            this.setSize([Math.max(size[0], 360), size[1]]);
            return result;
        };
    },
});
