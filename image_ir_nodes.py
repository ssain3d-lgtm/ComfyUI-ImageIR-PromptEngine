"""ComfyUI nodes for the IMAGE_IR prompt engine.

The graph is deliberately shaped so that the authoritative source cannot be
bypassed: IMAGE_IR is built by its own nodes, travels as its own wire type, and
is required by both the node that writes prompts and the node that checks them.
There is no input on the prompt engine that takes free text *about the image* —
the only free-text input it has is for directives that make no claim about the
picture (duration, aspect, render style), and the trace output marks them as
such so a reviewer can see which part of a prompt is not answerable to the IR.

The last node in any chain should be Image IR Grounding Guard. The engine
already audits itself, but a prompt that has been edited, concatenated with
someone's favourite style string, or written entirely by hand has no such
guarantee — and rule 9 asks for the check to happen on the final text.

The nodes here author, merge, inspect and compose from IMAGE_IR. Getting an
IMAGE_IR out of an actual image is image_ir_backend_nodes.py, and the MiniMax
H3 format lives there too.
"""

from __future__ import annotations

from .imageir.compose import compose
from .imageir.dsl import parse_dsl
from .imageir.guard import RULE_TEXT, audit
from .imageir.lexicon import MICRO_MOTIONS
from .imageir.motion import available_motions
from .imageir.schema import ImageIR, IRError, merge_ir, parse_ir

CATEGORY = "IMAGE_IR"

_MOTION_KEYS = tuple(str(m["key"]) for m in MICRO_MOTIONS)

_EXAMPLE_IR = """\
@frame viewer
@laterality unconfirmed

subject.identity      = a woman
subject.pose          = seated, torso upright
subject.gaze          = toward camera
subject.hands         = resting on the lap
subject.hair          = shoulder-length, dark
wardrobe.top          = blouse
wardrobe.top_material ~ smooth | satin, silk
scene.setting         = a plain studio backdrop
lighting.key          = soft light from the viewer-left
camera.shot           = medium shot
camera.height         = eye level
"""


def _as_ir(value) -> ImageIR:
    """Accept an IMAGE_IR wire value, a JSON string, or a mapping."""
    try:
        return parse_ir(value)
    except IRError as exc:
        raise ValueError(f"IMAGE_IR: {exc}") from exc


class ImageIRFromText:
    """Write IMAGE_IR by hand, in the line syntax or as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ir_text": ("STRING", {
                    "default": _EXAMPLE_IR, "multiline": True,
                    "tooltip": "One attribute per line: '=' observed, '~ hedge | candidates' uncertain, "
                               "'!' absent. Directives: @frame, @laterality, @note. JSON is accepted too.",
                }),
                "format": (["auto", "lines", "json"], {
                    "default": "auto",
                    "tooltip": "auto reads it as JSON when it starts with '{', otherwise as the line syntax.",
                }),
            },
            "optional": {
                "merge_into": ("IMAGE_IR", {
                    "tooltip": "Optional IMAGE_IR to extend. An observation is never overwritten by this "
                               "node's facts unless they observe the same attribute differently.",
                }),
                "on_conflict": (["error", "keep_base", "uncertain"], {
                    "default": "error",
                    "tooltip": "What to do when merge_into and this text observe one attribute differently. "
                               "uncertain collapses it back to uncertain, which is the only honest merge.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE_IR", "STRING")
    RETURN_NAMES = ("image_ir", "ir_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Author IMAGE_IR — the authoritative visual source every prompt in this graph is checked against."

    def build(self, ir_text, format="auto", merge_into=None, on_conflict="error"):
        text = (ir_text or "").strip()
        if not text:
            raise ValueError("IMAGE_IR: nothing to parse — the text is empty")
        use_json = format == "json" or (format == "auto" and text.startswith("{"))
        try:
            ir = parse_ir(text) if use_json else parse_dsl(text)
            if merge_into is not None:
                ir = merge_ir(_as_ir(merge_into), ir, on_conflict=on_conflict)
        except IRError as exc:
            raise ValueError(f"IMAGE_IR: {exc}") from exc
        return (ir, ir.to_json())


class ImageIRMerge:
    """Combine two IMAGE_IR documents without weakening either one's observations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE_IR",),
                "incoming": ("IMAGE_IR",),
                "on_conflict": (["error", "keep_base", "uncertain"], {
                    "default": "error",
                    "tooltip": "Two documents that observed one attribute differently cannot both be right. "
                               "error stops, keep_base ignores the incoming reading, uncertain marks the "
                               "attribute unresolved and records both readings as candidates.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE_IR", "STRING")
    RETURN_NAMES = ("image_ir", "ir_json")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Merge two IMAGE_IR documents. Sharpening is allowed; overwriting an observation is not."

    def run(self, base, incoming, on_conflict="error"):
        try:
            ir = merge_ir(_as_ir(base), _as_ir(incoming), on_conflict=on_conflict)
        except IRError as exc:
            raise ValueError(f"IMAGE_IR merge: {exc}") from exc
        return (ir, ir.to_json())


class ImageIRPromptEngine:
    """Compose CORE / DETAIL / FINAL prompts that contain nothing IMAGE_IR does not.

    This is the generic grounded composer. It is NOT a MiniMax H3 prompt — that
    format has its own node, and the tiers here were renamed away from H1/H2/H3
    precisely so the two can never be mistaken for each other.
    """

    @classmethod
    def INPUT_TYPES(cls):
        motions = {
            key: ("BOOLEAN", {
                "default": False,
                "tooltip": tooltip,
            })
            for key, tooltip in (
                ("blink", "Rule 6 micro-motion: a natural blink. Refused if IMAGE_IR records closed eyes."),
                ("breathing", "Rule 6 micro-motion: quiet breathing that barely moves the shoulders."),
                ("finger_movement", "Rule 6 micro-motion: the faintest movement in the fingers."),
                ("hair_movement", "Rule 6 micro-motion: a few strands of hair drifting."),
                ("weight_shift", "Rule 6 micro-motion: a barely perceptible settling of weight."),
            )
        }
        return {
            "required": {
                "image_ir": ("IMAGE_IR",),
                "style": (["sentence", "tags"], {
                    "default": "sentence",
                    "tooltip": "sentence adds the few structural connectives (wearing, in, lit by). "
                               "tags emits the recorded wordings comma-separated and nothing else.",
                }),
                "continuity": (["auto", "always", "never"], {
                    "default": "auto",
                    "tooltip": "Add 'gaze stays toward camera' wording for attributes a motion prompt tends "
                               "to drift on. auto adds it only when a micro-motion was accepted.",
                }),
                **motions,
                "require_motion_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: a micro-motion is only used when IMAGE_IR records the feature it names, "
                               "so rule 6 cannot smuggle in a fact rule 1 forbids. Off reads rule 6 literally.",
                }),
                "directives": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Non-visual directives only — duration, aspect, format, render style. "
                               "Kept verbatim and marked as directives in the trace. Anything here that "
                               "describes the image will be caught by the audit.",
                }),
                "on_violation": (["filter", "error", "report_only"], {
                    "default": "filter",
                    "tooltip": "What to do if the composed prompt fails its own audit. filter removes the "
                               "offending clauses, error stops the run, report_only passes the text through.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "core_prompt", "detail_prompt", "negative_prompt", "trace", "audit")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Build a grounded prompt strictly from IMAGE_IR, with only rule 6 micro-motions added."

    def run(
        self,
        image_ir,
        style="sentence",
        continuity="auto",
        blink=False,
        breathing=False,
        finger_movement=False,
        hair_movement=False,
        weight_shift=False,
        require_motion_anchor=True,
        directives="",
        on_violation="filter",
    ):
        ir = _as_ir(image_ir)
        requested = [
            key
            for key, enabled in (
                ("blink", blink),
                ("breathing", breathing),
                ("finger_movement", finger_movement),
                ("hair_movement", hair_movement),
                ("weight_shift", weight_shift),
            )
            if enabled
        ]
        composition = compose(
            ir,
            motions=requested,
            require_anchor=require_motion_anchor,
            continuity=continuity,
            directives=directives,
            style=style,
        )
        result = composition.self_audit
        prompt = composition.final
        if result is not None and not result.clean:
            if on_violation == "error":
                raise ValueError(
                    "IMAGE_IR grounding audit failed on the composed prompt:\n" + result.report()
                )
            if on_violation == "filter":
                prompt = result.filtered_prompt
        report = result.report() if result is not None else "no audit"
        return (prompt, composition.core, composition.detail, composition.negative, composition.trace(), report)


class ImageIRGroundingGuard:
    """Rule 9 — check a finished prompt against IMAGE_IR and strip what does not trace."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ir": ("IMAGE_IR",),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "The prompt to check. Every clause must trace to an IMAGE_IR fact, to the "
                               "rule 6 micro-motion allowlist, or to a directive that claims nothing visual.",
                }),
                "prompt_format": (["plain", "minimax_h3"], {
                    "default": "plain",
                    "tooltip": "minimax_h3 recognises the H3 document shape — field names, shot and picture "
                               "markers, the audio blocks — and audits only the visual description, so the "
                               "format survives the check. plain audits the whole text as prose.",
                }),
                "on_violation": (["filter", "error", "report_only"], {
                    "default": "filter",
                    "tooltip": "filter removes the offending clauses, error stops the run, report_only "
                               "passes the prompt through unchanged and reports what it found.",
                }),
                "allow_style_terms": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Let render-style wording through (cinematic, film grain, 85mm). Off requires "
                               "every word to describe something IMAGE_IR recorded.",
                }),
                "check_pronouns": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Treat a gendered pronoun as a claim about the subject, so it needs IMAGE_IR "
                               "to have recorded how the subject presents. Off allows 'she' / 'he' freely.",
                }),
                "require_motion_anchor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: a micro-motion in the prompt is only accepted when IMAGE_IR records the "
                               "feature it names.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("prompt", "audit", "violations", "clean")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Audit any prompt against IMAGE_IR and remove every statement that cannot be traced to it."

    def run(
        self,
        image_ir,
        prompt,
        prompt_format="plain",
        on_violation="filter",
        allow_style_terms=True,
        check_pronouns=True,
        require_motion_anchor=True,
    ):
        ir = _as_ir(image_ir)
        options = {
            "allow_style": allow_style_terms,
            "require_anchor": require_motion_anchor,
            "check_pronouns": check_pronouns,
        }
        if prompt_format == "minimax_h3":
            from .imageir.h3 import audit_h3_document

            result, _filtered = audit_h3_document(ir, prompt or "", **options)
        else:
            result = audit(ir, prompt or "", **options)
        if not result.clean and on_violation == "error":
            raise ValueError("IMAGE_IR grounding audit failed:\n" + result.report())
        text = result.filtered_prompt if (on_violation == "filter" and not result.clean) else (prompt or "")
        return (text, result.report(), len(result.findings), result.clean)


class ImageIRInspect:
    """Show what IMAGE_IR holds, and what it will and will not license."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_ir": ("IMAGE_IR",)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("ir_json", "summary", "available_motions")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Inspect an IMAGE_IR document: its facts by certainty, and the micro-motions it can carry."

    def run(self, image_ir):
        ir = _as_ir(image_ir)
        lines = [
            "IMAGE_IR summary",
            "=" * 16,
            f"geometry: {ir.frame}-relative, "
            f"subject laterality {'confirmed' if ir.subject_laterality_confirmed else 'NOT confirmed'}",
            "",
        ]
        for section in ir.sections():
            lines.append(f"{section}")
            for fact in ir.section(section):
                if fact.is_observed:
                    detail = f"= {fact.value}"
                elif fact.is_uncertain:
                    hedge = fact.hedge or "(no wording — omitted from prompts)"
                    candidates = f"  [never emit: {', '.join(fact.forbidden_specifics())}]" if fact.forbidden_specifics() else ""
                    detail = f"~ {hedge}{candidates}"
                else:
                    detail = "! absent"
                frame = f"  @{fact.frame}" if fact.frame else ""
                lines.append(f"  {fact.slot:<20} {detail}{frame}")
            lines.append("")
        if ir.notes:
            lines.append("notes (never composed into a prompt)")
            lines.extend(f"  {note}" for note in ir.notes)
            lines.append("")
        lines.append("grounding rules enforced")
        lines.append("-" * 16)
        lines.extend(f"  {number}. {text}" for number, text in sorted(RULE_TEXT.items()))

        motions = available_motions(ir)
        motion_text = ", ".join(motions) if motions else "(none — no IMAGE_IR anchor for any micro-motion)"
        summary = "\n".join(lines)
        # OUTPUT_NODE with a ui payload: the summary is the point of this node,
        # so it is shown on the node itself rather than only handed downstream.
        return {"ui": {"text": [summary]}, "result": (ir.to_json(), summary, motion_text)}


NODE_CLASS_MAPPINGS = {
    "ImageIRFromText": ImageIRFromText,
    "ImageIRMerge": ImageIRMerge,
    "ImageIRPromptEngine": ImageIRPromptEngine,
    "ImageIRGroundingGuard": ImageIRGroundingGuard,
    "ImageIRInspect": ImageIRInspect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageIRFromText": "Image IR Write (manual)",
    "ImageIRMerge": "Image IR Merge",
    "ImageIRPromptEngine": "Image IR Prompt Composer",
    "ImageIRGroundingGuard": "Image IR Grounding Guard",
    "ImageIRInspect": "Image IR Inspect",
}
