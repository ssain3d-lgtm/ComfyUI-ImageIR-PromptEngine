"""ComfyUI authoring, routing, reference and provenance sockets."""

from __future__ import annotations

from .imageir.backend import get_backend
from .imageir.backend.llama_cpp_launcher import LAUNCHER
from .imageir.h3_modes import compose_mode, route_mode
from .imageir.intent import MODES, author_intent, parse_intent
from .imageir.provenance import guard_document
from .imageir.references import ROLES, Reference, ReferencePack, parse_reference_pack
from .imageir.schema import parse_ir


def _socket(kind, tooltip):
    return (kind, {"tooltip": tooltip})


def _text(default="", tooltip="Authored text."):
    return ("STRING", {"default": default, "multiline": True, "tooltip": tooltip})


class H3UserIntentAuthor:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Translate Korean/English instructions into evidence-bearing USER_INTENT using the existing backend."
    FUNCTION = "run"
    RETURN_TYPES = ("USER_INTENT", "STRING")
    RETURN_NAMES = ("user_intent", "intent_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "backend_config": _socket("IMAGEIR_BACKEND", "Existing Image IR Backend Config."),
            "request": _text(tooltip="Freeform Korean or English video scene request."),
            "duration": ("FLOAT", {"default": 6.0, "min": 0.21, "max": 362 / 24, "tooltip": "Seconds; rounded up to the H3 frame grid."}),
            "mode_hint": (list(MODES), {"tooltip": "AUTO uses actual connected reference roles."}),
            "shot_count": ("INT", {"default": 1, "min": 1, "max": 24, "tooltip": "Exact shot count; use 1 unless cuts are desired."}),
        }, "optional": {name: _text(tooltip=f"Explicit {name} direction; translated as part of intent.")
                         for name in ("camera", "sound", "music", "style", "dialogue")}}

    def run(self, backend_config, request, duration=6.0, mode_hint="AUTO", shot_count=1,
            camera="", sound="", music="", style="", dialogue=""):
        if backend_config.provider == "llama_cpp" and backend_config.server_mode == "launch_local":
            LAUNCHER.start(backend_config)
        intent = author_intent(get_backend(backend_config), request, duration=duration,
                               mode_hint=mode_hint, shot_count=shot_count, camera=camera,
                               sound=sound, music=music, style=style, dialogue=dialogue)
        return intent, intent.to_json()


class H3UserIntentFromJSON:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Inspect, edit or import USER_INTENT_v1 without a network call."
    FUNCTION = "run"
    RETURN_TYPES = ("USER_INTENT", "STRING")
    RETURN_NAMES = ("user_intent", "intent_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"intent_json": _text(tooltip="A USER_INTENT_v1 JSON object with input evidence.")}}

    def run(self, intent_json):
        intent = parse_intent(intent_json)
        return intent, intent.to_json()


class H3ReferencePack:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Append one explicit image/video/audio asset role. Attach actual media downstream in matching label order."
    FUNCTION = "run"
    RETURN_TYPES = ("REFERENCE_PACK", "STRING")
    RETURN_NAMES = ("reference_pack", "reference_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "label": _text("Picture 1", "Asset label: Picture N, Video N or Audio N; match actual attachments."),
            "media_type": (["image", "video", "audio"], {"tooltip": "The referenced asset's media type."}),
            "role": (list(ROLES), {"tooltip": "Only facts supporting this role may enter the prompt."}),
            "subject": _text("1", "Subject number; use the same number when assets describe the same subject."),
        }, "optional": {
            "image_ir": _socket("IMAGE_IR", "Analyzer output for this image, if available."),
            "reference_pack": _socket("REFERENCE_PACK", "Previous pack; chain nodes for multiple references."),
        }}

    def run(self, label, media_type, role, subject="1", image_ir=None, reference_pack=None):
        pack = parse_reference_pack(reference_pack) if reference_pack is not None else ReferencePack()
        ref = Reference(label.strip(), media_type, role, subject.strip(), parse_ir(image_ir) if image_ir is not None else None)
        result = ReferencePack((*pack.references, ref))
        return result, result.to_json()


class H3ModeRouter:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Choose T2VA, I2VA, FL2VA, L2VA or Ref2VA. Reject ambiguous/mismatched inputs."
    FUNCTION = "run"
    RETURN_TYPES = ("H3_MODE", "STRING")
    RETURN_NAMES = ("mode", "routing_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mode": (list(MODES), {"tooltip": "Manual selection overrides intent hint and validates connected assets."})},
                "optional": {
                    "user_intent": _socket("USER_INTENT", "Optional structured intent mode hint."),
                    "first_ir": _socket("IMAGE_IR", "First/initial frame."),
                    "final_ir": _socket("IMAGE_IR", "Target/final frame."),
                    "reference_pack": _socket("REFERENCE_PACK", "Typed assets for Ref2VA."),
                }}

    def run(self, mode="AUTO", user_intent=None, first_ir=None, final_ir=None, reference_pack=None):
        hint = parse_intent(user_intent).mode_hint if user_intent is not None and mode == "AUTO" else mode
        pack = parse_reference_pack(reference_pack) if reference_pack is not None else None
        selected = route_mode(hint, first=first_ir, last=final_ir, references=pack)
        return selected, f"H3 mode: {selected}; input roles validated."


class MiniMaxH3ModeComposer:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Compose all five H3 modes from separate visual facts and USER_INTENT, retaining provenance."
    FUNCTION = "run"
    RETURN_TYPES = ("H3_DOCUMENT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("h3_document", "h3_prompt", "provenance_trace", "frames")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (list(MODES), {"tooltip": "Mode or AUTO; routed_mode socket overrides this choice."}),
            "user_intent": _socket("USER_INTENT", "Structured authoring state."),
        }, "optional": {
            "first_ir": _socket("IMAGE_IR", "Initial frame for I2VA/FL2VA."),
            "final_ir": _socket("IMAGE_IR", "Final frame for FL2VA/L2VA."),
            "reference_pack": _socket("REFERENCE_PACK", "Ref2VA role bindings."),
            "routed_mode": _socket("H3_MODE", "Connect H3 Mode Router; overrides the mode widget."),
        }}

    def run(self, mode, user_intent, first_ir=None, final_ir=None, reference_pack=None, routed_mode=None):
        result = compose_mode(user_intent, mode=routed_mode or mode, first=first_ir, last=final_ir, references=reference_pack)
        return result, result.render(), result.trace(), result.frames


class ImageIRProvenanceGuard:
    CATEGORY = "IMAGE_IR/H3"
    DESCRIPTION = "Validate H3 text against reconstructed IMAGE_IR / USER_INTENT / H3_RULE sources."
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_h3_prompt", "audit_report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"h3_document": _socket("H3_DOCUMENT", "Structured output from MiniMax H3 Mode Composer.")},
                "optional": {
                    "candidate": _text(tooltip="Optional text to check; blank checks the composed document."),
                    "on_violation": (["error", "filter"], {"tooltip": "Error or replace altered text with the validated composition."}),
                }}

    def run(self, h3_document, candidate="", on_violation="error"):
        if on_violation not in ("error", "filter"):
            raise ValueError("on_violation must be error or filter")
        result = guard_document(h3_document, candidate or None)
        if not result.clean and on_violation == "error":
            raise ValueError(result.report())
        return {"ui": {"text": [result.filtered_prompt, result.report()]},
                "result": (result.filtered_prompt, result.report())}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3UserIntentAuthor, H3UserIntentFromJSON, H3ReferencePack, H3ModeRouter, MiniMaxH3ModeComposer, ImageIRProvenanceGuard,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3UserIntentAuthor": "H3 User Intent Author",
    "H3UserIntentFromJSON": "H3 User Intent from JSON",
    "H3ReferencePack": "H3 Reference Pack",
    "H3ModeRouter": "H3 Mode Router",
    "MiniMaxH3ModeComposer": "MiniMax H3 Mode Composer (5 modes)",
    "ImageIRProvenanceGuard": "ImageIR Provenance Guard",
}
