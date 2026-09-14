"""Import helpers shared by the test modules.

The repository is a ComfyUI custom-node folder, so its name carries hyphens and
its node module uses package-relative imports. Tests load it the way ComfyUI
does — as a package — rather than reaching past the relative imports, so a
change that would break the import inside ComfyUI breaks the tests too.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_imageir_promptengine_under_test"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_node_package():
    """Load the repository root as a package, exactly as ComfyUI loads it."""
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the relative imports inside resolve
    # against this same module rather than starting a rival copy.
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module
