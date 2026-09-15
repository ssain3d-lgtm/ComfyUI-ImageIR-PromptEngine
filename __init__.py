"""ComfyUI entry point: the grounded IMAGE_IR prompt engine."""

from .image_ir_backend_nodes import NODE_CLASS_MAPPINGS as _BACKEND_NODES
from .image_ir_backend_nodes import NODE_DISPLAY_NAME_MAPPINGS as _BACKEND_NAMES
from .image_ir_intent_nodes import NODE_CLASS_MAPPINGS as _INTENT_NODES
from .image_ir_intent_nodes import NODE_DISPLAY_NAME_MAPPINGS as _INTENT_NAMES
from .image_ir_model_manager_nodes import NODE_CLASS_MAPPINGS as _MODEL_NODES
from .image_ir_model_manager_nodes import NODE_DISPLAY_NAME_MAPPINGS as _MODEL_NAMES
from .image_ir_nodes import NODE_CLASS_MAPPINGS as _CORE_NODES
from .image_ir_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES
from .image_ir_routes import register_routes

NODE_CLASS_MAPPINGS = {**_CORE_NODES, **_BACKEND_NODES, **_INTENT_NODES, **_MODEL_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_NAMES, **_BACKEND_NAMES, **_INTENT_NAMES, **_MODEL_NAMES}

# ComfyUI serves this directory under /extensions/<package>.  The JS adds the
# Connect/Refresh + model dropdown + Load/Unload controls to the manager node.
WEB_DIRECTORY = "./web"

# In unit tests / bare Python there is no PromptServer, so this is intentionally
# a no-op there. In ComfyUI it registers the same-origin proxy route used by the
# frontend without exposing API keys to browser JavaScript.
register_routes()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
