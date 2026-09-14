"""The MiniMax H3 I2VA prompt format, composed from IMAGE_IR.

This is a real H3 prompt, not the generic grounded prompt that compose.py
builds. The format comes from MiniMax's own prompt-writing guide, and the parts
that are load-bearing are reproduced exactly: the reference sentence that binds
the still to 0.00 seconds, the three field names, their order, the ``label:
value`` punctuation and the blank line between blocks. A field renamed or
reordered here is not a style difference — it is a prompt the model reads
differently.

Two things the format asks for that the IR cannot supply:

*Audio.* A still frame records no sound. overall_soundscape and
non_diegetic_music therefore come from the author, and they are not audited for
visual grounding — there is nothing visual in them to ground. ``N/A`` is a value
the format accepts, and it is the honest default.

*Camera motion.* How the camera moves over the next few seconds is not a fact
about the first frame either. It is chosen from a fixed list and written as
natural English inside the shot, which is what the guide asks for, rather than
stacked as a label at the end of a sentence. Subject motion and camera motion
are kept in separate sentences and separate fields on the result, because
conflating them is how a prompt ends up asking a subject to walk when it meant
the lens to move.

Everything else — what is in the frame at 0.00, and what stays true of it — is
IMAGE_IR's, and passes the same grounding guard as any other prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .compose import CORE_SLOTS, _continuity_clauses, _section_clauses
from .guard import AuditResult, audit_protected
from .motion import MotionPlan, MotionRejection, plan_motion
from .schema import ImageIR

# Reproduced verbatim from the MiniMax H3 prompt-writing reference. The I2VA
# case binds the supplied still to the start of the timeline with this sentence.
REFERENCE_LINE = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)

# Order is part of the format.
FIELD_ORDER = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")

SHOT_LABEL = "[Shot 1]"
NOT_APPLICABLE = "N/A"

# The format's own way of binding the description to the supplied still. A fixed
# string from the template, not prose about the image.
BINDING_PHRASE = "exactly as established in <Picture 1>"

# Syntax the format requires, which makes no claim about the image. Shielded
# from the grounding guard so the markers survive an audit that still judges
# every visual word around them.
H3_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(re.escape(BINDING_PHRASE)),
    re.compile(r"<Picture \d+>"),
    re.compile(r"<Subject \d+>"),
    re.compile(r"\[Shot \d+\]"),
    re.compile(r"\(S\d+(?:,\s*S\d+)*\)"),
    re.compile(r"\b\d+\.\d{2} seconds\b"),
    re.compile(r"\bN/A\b"),
)

# Written as an action inside the shot, per the guide, rather than as a trailing
# label. Viewer-relative throughout: the camera's left is the viewer's left, and
# the subject's is not knowable from a still.
CAMERA_MOTIONS: dict[str, str] = {
    "static": "The camera holds a locked-off framing throughout.",
    "slow_push_in": "The camera pushes in slowly across the shot.",
    "slow_pull_out": "The camera pulls out slowly across the shot.",
    "slow_pan_left": "The camera pans slowly toward the viewer-left.",
    "slow_pan_right": "The camera pans slowly toward the viewer-right.",
    "slow_tilt_up": "The camera tilts slowly upward.",
    "slow_tilt_down": "The camera tilts slowly downward.",
    "handheld_drift": "The camera drifts with a faint handheld float.",
}


@dataclass
class H3Prompt:
    """A composed I2VA prompt, its parts, and the audit of its visual content."""

    reference_line: str = REFERENCE_LINE
    integrated_multimodal_description: str = ""
    overall_soundscape: str = NOT_APPLICABLE
    non_diegetic_music: str = NOT_APPLICABLE

    # Kept apart so a caller can see which sentence is which, and so a test can
    # assert that subject motion never leaked into the camera clause.
    first_frame_anchor: str = ""
    continuity: str = ""
    subject_motion: str = ""
    camera_motion: str = ""

    motions: tuple[MotionPlan, ...] = ()
    rejected_motions: tuple[MotionRejection, ...] = ()
    audit_result: AuditResult | None = None
    unused_paths: tuple[str, ...] = ()
    ir_paths_used: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        """The prompt as MiniMax H3 expects to receive it."""
        blocks = [
            self.reference_line,
            f"integrated_multimodal_description: {self.integrated_multimodal_description}",
            f"overall_soundscape: {self.overall_soundscape}",
            f"non_diegetic_music: {self.non_diegetic_music}",
        ]
        return "\n\n".join(blocks)

    @property
    def clean(self) -> bool:
        return self.audit_result is None or self.audit_result.clean

    def trace(self) -> str:
        lines = ["MiniMax H3 (I2VA) composition", "=" * 29, ""]
        lines.append("visual content, grounded in IMAGE_IR")
        lines.append("-" * 29)
        for label, text in (
            ("first frame", self.first_frame_anchor),
            ("continuity", self.continuity),
            ("subject motion", self.subject_motion),
        ):
            lines.append(f"  {label:<15} {text or '—'}")
        lines.append("")
        lines.append("authored, not derived from the image")
        lines.append("-" * 29)
        lines.append(f"  {'camera motion':<15} {self.camera_motion or '—'}")
        lines.append(f"  {'soundscape':<15} {self.overall_soundscape}")
        lines.append(f"  {'music':<15} {self.non_diegetic_music}")
        if self.ir_paths_used:
            lines.append("")
            lines.append("IMAGE_IR facts used")
            lines.append("-" * 29)
            lines.extend(f"  {path}" for path in self.ir_paths_used)
        if self.rejected_motions:
            lines.append("")
            lines.append("motion refused")
            lines.append("-" * 29)
            lines.extend(f"  {r.key}: {r.reason}" for r in self.rejected_motions)
        if self.unused_paths:
            lines.append("")
            lines.append("IMAGE_IR facts not used")
            lines.append("-" * 29)
            lines.extend(f"  {path}" for path in self.unused_paths)
        return "\n".join(lines)


def document_shields(camera_sentence: str | None = None) -> tuple[re.Pattern[str], ...]:
    """Everything in a rendered H3 document that is not a claim about the image.

    The audio blocks are shielded whole, not word by word. overall_soundscape
    and non_diegetic_music describe sound, and IMAGE_IR describes a still —
    there is nothing in one to ground the other against, so auditing them for
    visual grounding would reject every honest soundscape ever written while
    catching nothing. Shielding them says that plainly, in one place, instead of
    scattering audio vocabulary through the guard's word lists where it would
    quietly start excusing visual claims too.
    """
    audio_block = re.compile(
        r"(?:overall_soundscape|non_diegetic_music):.*?(?=\n\n(?:integrated_multimodal_description|"
        r"overall_soundscape|non_diegetic_music):|\Z)",
        re.DOTALL,
    )
    camera = tuple(
        re.compile(re.escape(sentence))
        for sentence in ([camera_sentence] if camera_sentence else CAMERA_MOTIONS.values())
    )
    return (
        re.compile(re.escape(REFERENCE_LINE)),
        audio_block,
        re.compile(r"\b(?:integrated_multimodal_description|overall_soundscape|non_diegetic_music):"),
        *camera,
        *H3_MARKERS,
    )


def audit_h3_document(ir: ImageIR, text: str, **kwargs) -> tuple[AuditResult, str]:
    """Audit a rendered H3 prompt's visual content, leaving the format intact.

    Use this rather than the plain guard on anything in H3 shape: the plain
    guard has no reason to know that ``integrated_multimodal_description:`` is a
    field name, so it reads it as an untraceable claim and deletes the line that
    makes the prompt a prompt.
    """
    return audit_protected(ir, text, document_shields(), **kwargs)


def _sentence(text: str) -> str:
    text = text.strip().rstrip(",").strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _first_frame_anchor(ir: ImageIR, style: str) -> tuple[str, list[str]]:
    """What the picture shows at 0.00, in the IR's own words."""
    used: list[str] = []
    parts: list[str] = []

    core_slots = tuple(s for s in CORE_SLOTS if ir.get(f"subject.{s}"))
    groups = [("subject", _section_clauses(ir, "subject", core_slots, "sentence"))]
    rest = tuple(sorted({f.slot for f in ir.section("subject")} - set(core_slots)))
    groups.append(("subject", _section_clauses(ir, "subject", rest, "sentence")))
    for section in ir.sections():
        if section != "subject":
            groups.append((section, _section_clauses(ir, section, None, "sentence")))

    from .compose import SECTION_CONNECTIVES

    for section, clauses in groups:
        if not clauses:
            continue
        for clause in clauses:
            used.extend(clause.paths)
        body = ", ".join(c.text for c in clauses)
        connective = SECTION_CONNECTIVES.get(section, "")
        parts.append(f"{connective} {body}".strip() if connective else body)

    description = ", ".join(p for p in parts if p)
    lead = f"{SHOT_LABEL} "
    if style.strip():
        lead += f"{style.strip().rstrip(',')}, "
    anchor = f"{lead}{description}, {BINDING_PHRASE}" if description else f"{lead}<Picture 1>"
    return _sentence(anchor), used


def compose_h3(
    ir: ImageIR,
    *,
    motions: tuple[str, ...] | list[str] = (),
    require_anchor: bool = True,
    camera_motion: str = "static",
    overall_soundscape: str = "",
    non_diegetic_music: str = "",
    style: str = "",
    on_violation: str = "filter",
) -> H3Prompt:
    """Build a MiniMax H3 I2VA prompt whose visual content is entirely IMAGE_IR's.

    ``style`` is a non-visual directive ("Live-action, cinematic") that the
    official examples carry in the shot line. It is kept verbatim and reported
    as authored rather than derived, so a reader can see which words the image
    did not supply.
    """
    if camera_motion not in CAMERA_MOTIONS:
        raise ValueError(f"camera_motion must be one of {', '.join(CAMERA_MOTIONS)}, got {camera_motion!r}")
    if on_violation not in ("filter", "error", "report_only"):
        raise ValueError(f"on_violation must be filter, error or report_only, got {on_violation!r}")

    anchor, used_paths = _first_frame_anchor(ir, style)

    used: set[str] = set(used_paths)
    continuity_clauses = _continuity_clauses(ir, used)
    continuity = _sentence(", ".join(c.text for c in continuity_clauses))

    accepted, rejected = plan_motion(ir, tuple(motions), require_anchor=require_anchor)
    for plan in accepted:
        if plan.anchor:
            used.add(plan.anchor)
    subject_motion = _sentence(", ".join(plan.phrase for plan in accepted))

    camera_sentence = CAMERA_MOTIONS[camera_motion]

    description = " ".join(part for part in (anchor, continuity, subject_motion, camera_sentence) if part)

    # The audit runs on the visual description only. The audio fields describe
    # sound, which IMAGE_IR says nothing about either way, so auditing them for
    # visual grounding would reject every honest soundscape ever written.
    shields = H3_MARKERS + (re.compile(re.escape(camera_sentence)),)
    result, filtered = audit_protected(ir, description, shields, require_anchor=require_anchor)
    if not result.clean:
        if on_violation == "error":
            raise ValueError("the H3 visual description failed its grounding audit:\n" + result.report())
        if on_violation == "filter":
            description = filtered

    return H3Prompt(
        integrated_multimodal_description=description,
        overall_soundscape=(overall_soundscape.strip() or NOT_APPLICABLE),
        non_diegetic_music=(non_diegetic_music.strip() or NOT_APPLICABLE),
        first_frame_anchor=anchor,
        continuity=continuity,
        subject_motion=subject_motion,
        camera_motion=camera_sentence,
        motions=accepted,
        rejected_motions=rejected,
        audit_result=result,
        unused_paths=tuple(sorted(f.path for f in ir.facts if f.path not in used)),
        ir_paths_used=result.ir_paths_used,
    )
