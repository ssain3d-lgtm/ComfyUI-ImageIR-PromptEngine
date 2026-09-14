"""ComfyUI entry point: the grounded IMAGE_IR prompt engine.

Two node modules, merged here. image_ir_nodes holds everything that works on an
IMAGE_IR document already in hand; image_ir_backend_nodes holds everything that
reaches outside — vision models, local servers — plus the MiniMax H3 format.
"""

from .image_ir_backend_nodes import NODE_CLASS_MAPPINGS as _BACKEND_NODES
from .image_ir_backend_nodes import NODE_DISPLAY_NAME_MAPPINGS as _BACKEND_NAMES
from .image_ir_nodes import NODE_CLASS_MAPPINGS as _CORE_NODES
from .image_ir_nodes import NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES

NODE_CLASS_MAPPINGS = {**_CORE_NODES, **_BACKEND_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_NAMES, **_BACKEND_NAMES}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
