"""Rule 6 — the only motion a still frame can license.

A single image records a state, not a history. Any motion added to it is
invented, so the question is never "is this motion real?" but "could this
motion be true of the subject in this exact frame, without changing anything
the frame recorded?". Five motions pass that test: they are already happening
at some point in their cycle, and they leave gaze, pose, position and every
object where the image put them.

Two conditions gate each one:

*contradiction* — the IR may say something that rules the motion out outright.
Eyes recorded as closed cannot blink; hands recorded as out of frame have no
fingers to move.

*anchor* — the motion names a feature ("hair", "fingers"), and naming a feature
is itself a visual claim. With ``require_anchor`` on, a motion is only offered
when the IR records the feature it names, so rule 6 cannot be used as a side
door around rule 1. Off, rule 6 is read literally: any of the five, as long as
nothing contradicts it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexicon import MICRO_MOTIONS
from .schema import ImageIR


@dataclass(frozen=True)
class MotionPlan:
    """One accepted micro-motion and what in the IR let it through."""

    key: str
    phrase: str
    anchor: str | None  # the IR path the motion attaches to, if any

    @property
    def trace(self) -> str:
        return self.anchor or "rule 6 (micro-motion allowlist)"


@dataclass(frozen=True)
class MotionRejection:
    key: str
    reason: str


def _conflict_in_ir(ir: ImageIR, conflicts: tuple[str, ...]) -> tuple[str, str] | None:
    """The first IR fact whose own wording rules the motion out."""
    for fact in ir.facts:
        text = (fact.value or fact.hedge or "").lower()
        if not text:
            continue
        for phrase in conflicts:
            if phrase in text:
                return fact.path, phrase
    return None


def _anchor_in_ir(ir: ImageIR, slots: tuple[str, ...]) -> str | None:
    """The first anchor slot the IR actually records something for."""
    for path in slots:
        fact = ir.get(path)
        if fact is not None and not fact.is_absent:
            return path
    return None


def _absent_anchor(ir: ImageIR, slots: tuple[str, ...]) -> str | None:
    """An anchor slot the IR looked for and explicitly did not find."""
    for path in slots:
        fact = ir.get(path)
        if fact is not None and fact.is_absent:
            return path
    return None


def plan_motion(
    ir: ImageIR,
    requested: tuple[str, ...] | list[str],
    *,
    require_anchor: bool = True,
) -> tuple[tuple[MotionPlan, ...], tuple[MotionRejection, ...]]:
    """Decide which requested micro-motions this IR can carry.

    Anything not in the allowlist is rejected by name rather than dropped, so a
    request for "she raises her hand" leaves a trail in the report instead of
    quietly disappearing from the prompt.
    """
    catalogue = {str(m["key"]): m for m in MICRO_MOTIONS}
    accepted: list[MotionPlan] = []
    rejected: list[MotionRejection] = []
    seen: set[str] = set()

    for key in requested:
        key = str(key).strip()
        if not key or key in seen:
            continue
        seen.add(key)

        entry = catalogue.get(key)
        if entry is None:
            rejected.append(
                MotionRejection(key, "not in the rule 6 allowlist (blink, breathing, finger_movement, hair_movement, weight_shift)")
            )
            continue

        conflicts = tuple(entry["conflicts"])  # type: ignore[arg-type]
        clash = _conflict_in_ir(ir, conflicts)
        if clash is not None:
            path, phrase = clash
            rejected.append(MotionRejection(key, f"contradicts {path} ({phrase!r})"))
            continue

        slots = tuple(entry["slots"])  # type: ignore[arg-type]
        gone = _absent_anchor(ir, slots)
        if gone is not None:
            rejected.append(MotionRejection(key, f"{gone} is recorded as absent"))
            continue

        anchor = _anchor_in_ir(ir, slots)
        if anchor is None and require_anchor:
            rejected.append(
                MotionRejection(key, f"no IMAGE_IR anchor ({' or '.join(slots)}); naming it would add a visual fact (rule 1)")
            )
            continue

        accepted.append(MotionPlan(key=key, phrase=str(entry["phrase"]), anchor=anchor))

    return tuple(accepted), tuple(rejected)


def available_motions(ir: ImageIR, *, require_anchor: bool = True) -> tuple[str, ...]:
    """Every micro-motion this IR can carry, for surfacing the honest choices."""
    keys = [str(m["key"]) for m in MICRO_MOTIONS]
    accepted, _ = plan_motion(ir, keys, require_anchor=require_anchor)
    return tuple(plan.key for plan in accepted)
