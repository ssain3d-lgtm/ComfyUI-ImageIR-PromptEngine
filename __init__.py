"""ComfyUI entry point: the grounded IMAGE_IR prompt engine.

Node modules are merged here. The core package remains independent from ComfyUI;
these wrappers expose analysis, authoring, backend control, and llama.cpp router
model management to the graph.
"""

from .image_ir_backend_nodes import NODE_CLASS_MAPPINGS as _BACKEND_NODES
from .image_ir_backend_nodes import NODE_DISPLAY_NAME_MAPPINGS as _BACKEND_NAMES
from .image_ir_intent_nodes import NODE_CLASS_MAPPINGS as _INTENT_NODES
from .image_ir_intent_nodes import NODE_DISPLAY_NAME_MAPPINGS as _INTENT_NAMES
from .image_ir_llama_manager_nodes import NODE_CLASS_MAPPINGS as _LLAMA_MANAGER_NODES
from .image_ir_llama_manager_nodes import NODE_DISPLAY_NAME_MAPPINGS as _LLAMA_MANAGER_NAMES
from .image_ir_nodes import NODE_CLASS_MAPPINGS as _CORE_NODES
from .image_ir_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES

NODE_CLASS_MAPPINGS = {
    **_CORE_NODES,
    **_BACKEND_NODES,
    **_INTENT_NODES,
    **_LLAMA_MANAGER_NODES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_CORE_NAMES,
    **_BACKEND_NAMES,
    **_INTENT_NAMES,
    **_LLAMA_MANAGER_NAMES,
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
