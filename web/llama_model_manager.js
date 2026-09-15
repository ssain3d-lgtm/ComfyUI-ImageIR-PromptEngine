import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function widget(node, name) {
    return node.widgets?.find((entry) => entry.name === name);
}

function setComboValues(combo, models, preferred) {
    const ids = models.map((model) => model.id);
    combo.options = combo.options || {};
    combo.options.values = ids.length ? ids : [""];
    if (preferred && ids.includes(preferred)) {
        combo.value = preferred;
    } else if (!ids.includes(combo.value)) {
        combo.value = ids[0] || "";
    }
}

async function routerAction(node, action) {
    const baseUrl = widget(node, "base_url")?.value || "http://127.0.0.1:8080";
    const selected = widget(node, "selected_model");
    const tokenEnv = widget(node, "api_token_env")?.value || "";
    const timeout = widget(node, "timeout")?.value || 120;
    const response = await api.fetchApi("/imageir/llama-router", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            base_url: baseUrl,
            selected_model: selected?.value || "",
            action,
            api_token_env: tokenEnv,
            timeout,
        }),
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || `llama.cpp router request failed (${response.status})`);
    }
    setComboValues(selected, data.models || [], data.selected_model);
    const loaded = (data.models || []).filter((model) => model.status === "loaded").map((model) => model.id);
    const vision = (data.models || []).filter((model) => model.vision).map((model) => model.id);
    node.title = loaded.length ? `ImageIR llama.cpp Manager · ${loaded.join(", ")}` : "ImageIR llama.cpp Model Manager";
    node.imageirRouterStatus = `${data.message}\nLoaded: ${loaded.join(", ") || "none"}\nVision: ${vision.join(", ") || "none"}`;
    node.setDirtyCanvas(true, true);
    return data;
}

app.registerExtension({
    name: "ImageIR.LlamaModelManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ImageIRLlamaModelManager") return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.addWidget("button", "Connect / Refresh", null, async () => {
                try {
                    await routerAction(this, "refresh");
                } catch (error) {
                    console.error("ImageIR llama manager:", error);
                    this.title = `ImageIR llama.cpp Manager · ${error.message}`;
                    this.setDirtyCanvas(true, true);
                }
            });
            this.addWidget("button", "Load", null, async () => {
                try {
                    await routerAction(this, "load");
                } catch (error) {
                    console.error("ImageIR llama manager:", error);
                }
            });
            this.addWidget("button", "Unload", null, async () => {
                try {
                    await routerAction(this, "unload");
                } catch (error) {
                    console.error("ImageIR llama manager:", error);
                }
            });
            return result;
        };
    },
});
