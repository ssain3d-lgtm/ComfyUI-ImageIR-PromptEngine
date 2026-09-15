"""Validate source boundaries and reject edits that cannot be traced to them."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .guard import _check_contradiction, audit
from .schema import ImageIR


@dataclass(frozen=True)
class Provenance:
    source: str
    path: str
    text: str
    evidence: str = ""


@dataclass(frozen=True)
class ProvenanceAudit:
    findings: tuple[str, ...]
    filtered_prompt: str

    @property
    def clean(self):
        return not self.findings

    def report(self):
        return "Provenance guard: clean" if self.clean else "\n".join(self.findings)


def validate_boundaries(intent, first=None, last=None):
    for name, ir in (("start_states", first), ("final_states", last)):
        if ir is None:
            continue
        for clause in getattr(intent, name):
            fact = ir.get(clause.path)
            if not fact or not fact.emitted_text():
                raise ValueError(f"{name}: {clause.path} is not established in IMAGE_IR; review requested boundary")
            scoped = ImageIR((fact,), ir.frame, ir.subject_laterality_confirmed)
            if not audit(scoped, clause.text, require_anchor=False, check_pronouns=False).clean:
                raise ValueError(f"{name}: requested {clause.path} conflicts with or exceeds IMAGE_IR")
    visual_fields = ("summary", "requested_actions", "requested_scene_progression", "requested_camera", "requested_style", "constraints")
    for clause in (c for name in visual_fields for c in getattr(intent, name)):
        # Explicit terminal claims cannot override the provided last frame, even
        # when the author placed them in actions/camera instead of final_states.
        ending = re.search(r"\b(?:finally|at the end|ends? (?:with|in)|final (?:frame|state))\b(.+)", clause.text, re.I)
        if last is not None and ending:
            text = re.sub(r"\bsits? down\b", "seated", ending.group(1), flags=re.I)
            if _check_contradiction(last, text):
                raise ValueError("requested final state contradicts the final IMAGE_IR boundary")
        if first is None or clause.shot != 1:
            continue
        prior = re.search(r"\b(?:from (?:initially )?|initially |at (?:the )?start[, ]+|starts? (?:out )?|begins? )"
                          r"(.+?)(?:[.;]| (?:and )?then |$)", clause.text, re.I)
        if prior and _check_contradiction(first, prior.group(1)):
            raise ValueError("requested prior/start state contradicts IMAGE_IR")
        pose = first.get("subject.pose")
        if pose and pose.is_observed:
            low = clause.text.lower()
            observed = (pose.value or "").lower()
            if (re.search(r"\b(?:stands?|rises?|gets?) (?:up |upwards )?from\b", low)
                    and "standing" in observed):
                raise ValueError("start pose is already standing; cannot invent a seated prior state")


def guard_document(document, candidate: str | None = None) -> ProvenanceAudit:
    """Rebuild from source objects; never trust edited sections or trace labels.

    This intentionally requires exact deterministic composition. To edit a prompt,
    edit its intent/IR and compose again. A free-text addition, even an audio or
    camera addition, cannot borrow the authority of a neighbouring clause.
    """
    from .h3_modes import compose_mode

    expected = compose_mode(document.intent, mode=document.mode, first=document.first,
                            last=document.last, references=document.references)
    text = document.render() if candidate is None else candidate
    if text != expected.render():
        return ProvenanceAudit(("Untraced or altered prompt content; update USER_INTENT or IMAGE_IR and recompose.",), expected.render())
    return ProvenanceAudit((), text)
