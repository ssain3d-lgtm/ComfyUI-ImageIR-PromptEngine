"""Deterministic five-mode H3 formatting with auditable source clauses."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .h3 import FIELD_ORDER, REFERENCE_LINE
from .intent import MODES, UserIntent, parse_intent
from .provenance import Provenance, validate_boundaries
from .references import ReferencePack, parse_reference_pack
from .schema import ImageIR, parse_ir

REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
              "overall_soundscape", "non_diegetic_music")


def route_mode(mode="AUTO", *, first=None, last=None, references=None):
    if mode not in MODES:
        raise ValueError("unknown H3 mode")
    has_refs = bool(references and references.references)
    if has_refs and (first is not None or last is not None):
        raise ValueError("ambiguous frame anchors and typed references; select a single input family")
    inferred = "Ref2VA" if has_refs else ("FL2VA" if first is not None and last is not None else
                "I2VA" if first is not None else "L2VA" if last is not None else "T2VA")
    if mode != "AUTO" and mode != inferred:
        raise ValueError(f"{mode} inputs are incompatible; supplied inputs describe {inferred}")
    return inferred if mode == "AUTO" else mode


def grid_frames(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 362 / 24:
        raise ValueError("duration must be >0 and <=362/24 seconds; split longer requests")
    return max(5, 17 * math.ceil((seconds * 24 - 5) / 17 - 1e-12) + 5)


def _timestamp(seconds):
    millis = round(seconds * 1000)
    return f"{millis // 60000:02d}:{(millis // 1000) % 60:02d}.{millis % 1000:03d}"


@dataclass(frozen=True)
class ModePrompt:
    mode: str
    frames: int
    alignment: str
    sections: tuple[tuple[str, str], ...]
    provenance: tuple[Provenance, ...]
    warnings: tuple[str, ...]
    intent: UserIntent
    first: ImageIR | None
    last: ImageIR | None
    references: ReferencePack

    def render(self):
        return "\n\n".join(([self.alignment] if self.alignment else []) +
                            [f"{name}: {body}" for name, body in self.sections])

    def trace(self):
        lines = [f"H3 {self.mode}; length {self.frames} ({self.frames / 24:.2f} s)"]
        lines.extend(f"{p.source} {p.path}: {p.text}" + (f" <- {p.evidence}" if p.evidence else "") for p in self.provenance)
        lines.extend("WARNING: " + w for w in self.warnings)
        return "\n".join(lines)


def compose_mode(intent, *, mode="AUTO", first=None, last=None, references=None) -> ModePrompt:
    intent = parse_intent(intent)
    first = parse_ir(first) if first is not None else None
    last = parse_ir(last) if last is not None else None
    references = parse_reference_pack(references) if references is not None else ReferencePack()
    selected = route_mode(intent.mode_hint if mode == "AUTO" else mode, first=first, last=last, references=references)
    validate_boundaries(intent, first, last)
    frames = grid_frames(intent.duration_hint)
    if intent.shot_count_hint > frames:
        raise ValueError("shot count exceeds video frame count")
    duration = frames / 24
    count = intent.shot_count_hint
    trace = []
    warnings = ["USER_INTENT translation is model-authored; evidence spans are checked, semantic fidelity needs review."]

    def rule(text, path="format"):
        trace.append(Provenance("H3_RULE", path, text))
        return text

    def visual(ir, label):
        parts = []
        for fact in ir.facts:
            text = fact.emitted_text()
            if text:
                parts.append(text)
                trace.append(Provenance("IMAGE_IR", f"{label}.{fact.path}", text, fact.evidence or ""))
            if fact.verification_required or (fact.confidence is not None and fact.confidence < 0.6):
                warnings.append(f"Review {label}.{fact.path}: low confidence or verification_required")
        return ", ".join(parts)

    def authored(name, shot=None):
        parts = []
        for i, clause in enumerate(getattr(intent, name)):
            if shot is not None and clause.shot != shot:
                continue
            text = clause.text
            trace.append(Provenance("USER_INTENT", f"{name}[{i}]", text,
                                    f"inputs.{clause.input}: {clause.evidence}"))
            if name == "dialogue":
                language = "Korean" if any("\uac00" <= c <= "\ud7a3" for c in text) else "English"
                text = rule(f"(S1) says: <d>[{language}] ", "dialogue") + text + rule("</d>", "dialogue")
            parts.append(text)
        return " ".join(parts)

    alignment = ""
    if selected == "I2VA":
        alignment = rule(REFERENCE_LINE, "alignment")
    elif selected == "FL2VA":
        alignment = rule("How the reference pictures align with the target video — Picture 1 (from Shot 1) "
                         "aligns with the 0.00-second mark of the target video; "
                         f"Picture 2 (from Shot {count}) aligns with the {duration:.2f}-second mark of the target video.", "alignment")
    elif selected == "L2VA":
        alignment = rule("How the reference pictures align with the target video — "
                         f"<Picture 1> (from [Shot {count}]) aligns with the {duration:.2f}-second mark of the target video.", "alignment")

    definitions, retention = [], []
    if selected == "Ref2VA":
        groups = {}
        for ref in references.references:
            groups.setdefault(ref.subject, []).append(ref)
        for subject, refs in groups.items():
            parts = []
            for ref in refs:
                role = "wardrobe" if ref.role == "clothing" else ref.role
                contribution = f"{role} from <{ref.label}>"
                trace.append(Provenance("USER_INTENT", f"REFERENCE_PACK.{ref.label}.role", contribution))
                detail = visual(ref.scoped_ir(), ref.label)
                parts.append(contribution + (f" ({detail})" if detail else ""))
                if ref.type != "image" or role == "composition":
                    marker = "reference" if ref.type == "audio" else "weak_reference"
                    retention.append(rule(f"<{ref.label}> ({role}): {marker}.", "retention"))
                if ref.type == "image" and ref.image_ir is None:
                    warnings.append(f"{ref.label} has a role but no analyzed IMAGE_IR; no visual details inferred")
            definitions.append(rule(f"<Subject {subject}>: ", "subject") + "; ".join(parts) + ".")
            retention.insert(0, rule(f"<Subject {subject}>: attribute_transfer — assigned asset roles.", "retention"))

    style = authored("requested_style")
    shots = []
    for shot in range(1, count + 1):
        marker = f"[Shot {shot}]"
        if shot > 1:
            marker += f" At {_timestamp((frames * (shot - 1) // count) / 24)},"
        parts = [rule(marker, "timeline")]
        if shot == 1:
            if selected != "Ref2VA" and style:
                parts.append(style)
            if first is not None:
                parts.extend([rule("At the first frame,", "anchor"), visual(first, "first"),
                              rule("exactly as established in <Picture 1>.", "anchor")])
            if selected != "Ref2VA":
                parts.append(authored("summary"))
            if first is None and intent.start_states:
                parts.extend([rule("At the first frame,", "authored_start"), authored("start_states")])
        for name in ("requested_actions", "requested_scene_progression", "requested_camera", "dialogue"):
            parts.append(authored(name, shot))
        if shot == count and last is not None:
            picture = 2 if first is not None else 1
            parts.extend([rule(f"At {duration:.2f} seconds, the final frame shows", "landing"), visual(last, "last"),
                          rule(f"exactly as established in <Picture {picture}>.", "landing")])
        elif shot == count and intent.final_states:
            parts.extend([rule("At the final frame,", "authored_landing"), authored("final_states")])
        shots.append(" ".join(p for p in parts if p))
    constraints = authored("constraints")
    body = " ".join(shots) + (" " + constraints if constraints else "")
    sound = authored("requested_sound") or rule("N/A", "audio_default")
    music = authored("requested_music") or rule("N/A", "audio_default")
    if selected == "Ref2VA":
        sections = tuple(zip(REF_FIELDS, ("\n".join(definitions),
            rule("[reference generation]", "task_type") + " " + (authored("summary") or authored("requested_actions")),
            "\n".join(retention), (style + " " if style else "") + body, sound, music), strict=True))
    else:
        sections = tuple(zip(FIELD_ORDER, (body, sound, music), strict=True))
    for name, _ in sections:
        rule(name + ":", "section")
    return ModePrompt(selected, frames, alignment, sections, tuple(trace), tuple(warnings), intent, first, last, references)
