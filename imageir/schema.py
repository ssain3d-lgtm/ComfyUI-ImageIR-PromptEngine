"""IMAGE_IR — the authoritative record of what an image actually shows.

The whole package exists to enforce one asymmetry: IMAGE_IR is written once,
from the image, and every stage downstream may only *read* it. A composer may
drop a fact, hedge it or reorder it. It may never add one, sharpen one, or
change one. That rule is not a style preference; a prompt that invents a
visual fact silently replaces the reference image with something else.

Three certainty levels carry that asymmetry through the pipeline:

``observed``
    The attribute was read off the image. Its value is emitted verbatim and no
    stage may alter it.
``uncertain``
    Something is there but the specific value could not be read. Only the
    generic ``hedge`` may be emitted ("smooth blouse"), never the candidates
    that were considered ("satin blouse"). Resolving an uncertain attribute is
    the most tempting failure mode in prompt writing, so the candidates are
    kept in the IR *specifically* so the guard can recognise and reject them.
``absent``
    The attribute was looked for and is not in the image. Nothing is emitted,
    and any mention of it in a prompt is a grounded-fact violation.

Geometry gets the same treatment. An IR written from a photograph knows only
viewer-relative sides ("the hand on the viewer's left"). Turning that into
subject-relative sides ("her left hand") is a guess about which way the
subject faces, so it is allowed only when the IR explicitly confirms subject
laterality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

IR_VERSION = "1.0"

# Read off the image / present but unreadable / looked for and not there.
OBSERVED = "observed"
UNCERTAIN = "uncertain"
ABSENT = "absent"
CERTAINTIES = (OBSERVED, UNCERTAIN, ABSENT)

# Whose left is "left". An IR written from a still frame is viewer-relative
# unless the writer confirms which way the subject faces.
VIEWER = "viewer"
SUBJECT = "subject"
FRAMES = (VIEWER, SUBJECT)

# Sections exist to give the composer a sentence order, not to constrain the
# IR: an unknown section is kept and simply composed after the known ones.
SECTION_ORDER = ("subject", "wardrobe", "scene", "lighting", "camera")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no visual claim of their own. Kept out of the vocabulary a
# prompt is traced against so that "a woman with a calm expression" is not
# judged grounded by "a", "with" and "the" alone.
#
# Two kinds of word are deliberately NOT here. Directional prepositions —
# toward, into, over, behind — are half of what a spatial observation says, so
# "gaze toward camera" and "gaze over the shoulder" must not tokenize alike.
# Gendered pronouns are a claim about the subject, so they are left in for the
# guard to check; the neutral ones (they, it) assert nothing and stay.
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have here in is it
    its of off on or our that the their theirs them then there these they this
    those to up was were what when where which while who whom whose will with
    you your
    """.split()
)


class IRError(ValueError):
    """A malformed IMAGE_IR document. The message names the offending path."""


def tokenize(text: str | None) -> tuple[str, ...]:
    """Lowercase content words, punctuation and stopwords dropped."""
    if not text:
        return ()
    return tuple(t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS)


@dataclass(frozen=True)
class Fact:
    """One attribute of the image, with how well it is known."""

    section: str
    slot: str
    value: str | None
    certainty: str
    hedge: str | None = None
    # Specific readings that were considered and could not be confirmed. Never
    # emitted; carried so the guard can reject a prompt that picks one.
    candidates: tuple[str, ...] = ()
    # Per-fact override of the document's geometry frame, for a fact whose
    # laterality is known even though the rest of the IR's is not.
    frame: str | None = None

    @property
    def path(self) -> str:
        return f"{self.section}.{self.slot}"

    @property
    def is_observed(self) -> bool:
        return self.certainty == OBSERVED

    @property
    def is_uncertain(self) -> bool:
        return self.certainty == UNCERTAIN

    @property
    def is_absent(self) -> bool:
        return self.certainty == ABSENT

    def emitted_text(self) -> str | None:
        """The only wording this fact licenses in a prompt.

        Observed facts are emitted exactly as recorded. An uncertain fact is
        emitted through its hedge or not at all — never through its value,
        which is the unconfirmed reading. An absent fact is emitted never.
        """
        if self.is_observed:
            return self.value or None
        if self.is_uncertain:
            return self.hedge or None
        return None

    def forbidden_specifics(self) -> tuple[str, ...]:
        """Wordings that would resolve this fact beyond what the image shows."""
        if not self.is_uncertain:
            return ()
        specifics = list(self.candidates)
        if self.value:
            specifics.append(self.value)
        seen: dict[str, None] = {}
        for item in specifics:
            text = item.strip()
            if text and text != self.hedge:
                seen[text] = None
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "certainty": self.certainty}
        if self.hedge:
            out["hedge"] = self.hedge
        if self.candidates:
            out["candidates"] = list(self.candidates)
        if self.frame:
            out["frame"] = self.frame
        return out


@dataclass(frozen=True)
class ImageIR:
    """A parsed, normalised IMAGE_IR document."""

    facts: tuple[Fact, ...] = ()
    frame: str = VIEWER
    subject_laterality_confirmed: bool = False
    # Free text about provenance or capture conditions. Never composed into a
    # prompt and never counted as grounding: a note is not an observation.
    notes: tuple[str, ...] = ()
    ir_version: str = IR_VERSION

    def get(self, path: str) -> Fact | None:
        for fact in self.facts:
            if fact.path == path:
                return fact
        return None

    def section(self, name: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.section == name)

    def sections(self) -> tuple[str, ...]:
        """Section names in composition order, unknown sections last."""
        present = {f.section for f in self.facts}
        known = [s for s in SECTION_ORDER if s in present]
        extra = sorted(present - set(SECTION_ORDER))
        return tuple(known + extra)

    def observed(self) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.is_observed)

    def emittable(self) -> tuple[Fact, ...]:
        """Facts that license some wording — observed, or uncertain with a hedge."""
        return tuple(f for f in self.facts if f.emitted_text())

    def frame_for(self, fact: Fact) -> str:
        return fact.frame or self.frame

    def vocabulary(self) -> frozenset[str]:
        """Every content word a prompt may draw on and still be traceable.

        Slot names are included alongside values: an IR that records
        ``subject.gaze = toward camera`` grounds the word "gaze" as much as it
        grounds "camera". Candidates are excluded — they are exactly the words
        a prompt must not reach for.
        """
        words: set[str] = set()
        for fact in self.facts:
            text = fact.emitted_text()
            if text:
                words.update(tokenize(text))
                words.update(tokenize(fact.slot.replace("_", " ")))
        return frozenset(words)

    def trace_index(self) -> dict[str, tuple[str, ...]]:
        """Content word -> the IR paths that license it."""
        index: dict[str, set[str]] = {}
        for fact in self.facts:
            text = fact.emitted_text()
            if not text:
                continue
            for word in tokenize(text) + tokenize(fact.slot.replace("_", " ")):
                index.setdefault(word, set()).add(fact.path)
        return {word: tuple(sorted(paths)) for word, paths in index.items()}

    def with_fact(self, fact: Fact) -> ImageIR:
        """A copy with ``fact`` added or replacing the one at the same path."""
        kept = [f for f in self.facts if f.path != fact.path]
        kept.append(fact)
        return replace(self, facts=tuple(kept))

    def to_dict(self) -> dict[str, Any]:
        sections: dict[str, dict[str, Any]] = {}
        for fact in self.facts:
            sections.setdefault(fact.section, {})[fact.slot] = fact.to_dict()
        out: dict[str, Any] = {
            "ir_version": self.ir_version,
            "geometry": {
                "frame": self.frame,
                "subject_laterality_confirmed": self.subject_laterality_confirmed,
            },
            "sections": sections,
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=False)


def _parse_fact(section: str, slot: str, raw: Any) -> Fact:
    """Accept the full form or one of the hand-authoring shorthands.

    ``"toward camera"``   observed
    ``"~smooth"``         uncertain, hedged as "smooth" (bare ``"~"``: no wording at all)
    ``None``              absent
    ``{...}``             the full form, the only one that can carry candidates
    """
    path = f"{section}.{slot}"
    if raw is None:
        return Fact(section=section, slot=slot, value=None, certainty=ABSENT)

    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("~"):
            hedge = text[1:].strip()
            return Fact(section=section, slot=slot, value=None, certainty=UNCERTAIN, hedge=hedge or None)
        if not text:
            return Fact(section=section, slot=slot, value=None, certainty=ABSENT)
        return Fact(section=section, slot=slot, value=text, certainty=OBSERVED)

    if not isinstance(raw, dict):
        raise IRError(f"{path}: expected a string, null or an object, got {type(raw).__name__}")

    unknown = set(raw) - {"value", "certainty", "hedge", "candidates", "frame"}
    if unknown:
        raise IRError(f"{path}: unknown key(s) {', '.join(sorted(unknown))}")

    value = raw.get("value")
    if value is not None and not isinstance(value, str):
        raise IRError(f"{path}: value must be a string or null")
    value = value.strip() if isinstance(value, str) else None

    certainty = raw.get("certainty")
    if certainty is None:
        # An object that only carries a hedge is describing an unreadable
        # attribute; anything else with a value is an observation.
        certainty = UNCERTAIN if raw.get("hedge") or raw.get("candidates") else (OBSERVED if value else ABSENT)
    if certainty not in CERTAINTIES:
        raise IRError(f"{path}: certainty must be one of {', '.join(CERTAINTIES)}, got {certainty!r}")

    hedge = raw.get("hedge")
    if hedge is not None and not isinstance(hedge, str):
        raise IRError(f"{path}: hedge must be a string")
    hedge = hedge.strip() if isinstance(hedge, str) else None

    candidates_raw = raw.get("candidates") or ()
    if isinstance(candidates_raw, str):
        candidates_raw = [candidates_raw]
    if not isinstance(candidates_raw, (list, tuple)):
        raise IRError(f"{path}: candidates must be a list of strings")
    candidates = tuple(c.strip() for c in candidates_raw if isinstance(c, str) and c.strip())

    frame = raw.get("frame")
    if frame is not None and frame not in FRAMES:
        raise IRError(f"{path}: frame must be one of {', '.join(FRAMES)}, got {frame!r}")

    if certainty == OBSERVED and not value:
        raise IRError(f"{path}: an observed fact needs a value")
    if certainty == ABSENT and value:
        raise IRError(f"{path}: an absent fact cannot carry a value")

    return Fact(
        section=section,
        slot=slot,
        value=value,
        certainty=certainty,
        hedge=hedge,
        candidates=candidates,
        frame=frame,
    )


def _parse_sections(raw: Any) -> list[Fact]:
    if not isinstance(raw, dict):
        raise IRError("sections must be an object of section name -> attributes")
    facts: list[Fact] = []
    for section, slots in raw.items():
        if not isinstance(slots, dict):
            raise IRError(f"{section}: expected an object of attribute -> value")
        for slot, value in slots.items():
            facts.append(_parse_fact(str(section), str(slot), value))
    return facts


def parse_ir(source: Any) -> ImageIR:
    """Parse an IMAGE_IR document from JSON text, a mapping, or an ``ImageIR``.

    Unknown top-level keys are an error rather than a shrug: a typo in
    ``subject_laterality_confirmed`` would otherwise silently leave the safe
    default in place while the author believes laterality was confirmed.
    """
    if isinstance(source, ImageIR):
        return source

    if isinstance(source, (str, bytes)):
        text = source.decode() if isinstance(source, bytes) else source
        if not text.strip():
            raise IRError("IMAGE_IR is empty")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise IRError(f"IMAGE_IR is not valid JSON: {exc}") from exc
    else:
        data = source

    if not isinstance(data, dict):
        raise IRError(f"IMAGE_IR must be an object, got {type(data).__name__}")

    unknown = set(data) - {"ir_version", "geometry", "sections", "notes"}
    if unknown:
        raise IRError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")

    geometry = data.get("geometry") or {}
    if not isinstance(geometry, dict):
        raise IRError("geometry must be an object")
    unknown_geo = set(geometry) - {"frame", "subject_laterality_confirmed"}
    if unknown_geo:
        raise IRError(f"geometry: unknown key(s) {', '.join(sorted(unknown_geo))}")

    frame = geometry.get("frame", VIEWER)
    if frame not in FRAMES:
        raise IRError(f"geometry.frame must be one of {', '.join(FRAMES)}, got {frame!r}")

    confirmed = geometry.get("subject_laterality_confirmed", False)
    if not isinstance(confirmed, bool):
        raise IRError("geometry.subject_laterality_confirmed must be true or false")

    notes_raw = data.get("notes") or ()
    if isinstance(notes_raw, str):
        notes_raw = [notes_raw]
    if not isinstance(notes_raw, (list, tuple)):
        raise IRError("notes must be a string or a list of strings")
    notes = tuple(str(n).strip() for n in notes_raw if str(n).strip())

    facts = _parse_sections(data.get("sections") or {})

    return ImageIR(
        facts=tuple(facts),
        frame=frame,
        subject_laterality_confirmed=confirmed,
        notes=notes,
        ir_version=str(data.get("ir_version") or IR_VERSION),
    )


class MergeConflict(IRError):
    """Two IR documents disagree about an attribute both claim to have observed."""


def merge_ir(base: ImageIR, incoming: ImageIR, *, on_conflict: str = "error") -> ImageIR:
    """Combine two IR documents without ever overwriting an observation.

    Sharpening is the one direction that is always safe: an absent or uncertain
    fact may be replaced by an observation of the same attribute, because the
    observation is the stronger reading of the same image. The reverse is not,
    and two different observations of one attribute cannot both be true —
    ``on_conflict`` decides whether that raises or collapses the attribute back
    to uncertain, which is the only merge that keeps the result honest.
    """
    if on_conflict not in ("error", "keep_base", "uncertain"):
        raise IRError(f"on_conflict must be error, keep_base or uncertain, got {on_conflict!r}")

    merged = base
    for fact in incoming.facts:
        current = merged.get(fact.path)
        if current is None:
            merged = merged.with_fact(fact)
            continue
        if current.is_observed and fact.is_observed and current.value != fact.value:
            if on_conflict == "error":
                raise MergeConflict(
                    f"{fact.path}: both documents observed this attribute but disagree "
                    f"({current.value!r} vs {fact.value!r})"
                )
            if on_conflict == "keep_base":
                continue
            merged = merged.with_fact(
                replace(
                    current,
                    value=None,
                    certainty=UNCERTAIN,
                    hedge=None,
                    candidates=tuple(dict.fromkeys((current.value or "", fact.value or ""))),
                )
            )
            continue
        if current.is_observed and not fact.is_observed:
            continue  # never weaken an observation
        merged = merged.with_fact(fact)

    laterality = base.subject_laterality_confirmed and incoming.subject_laterality_confirmed
    frame = base.frame if base.frame == incoming.frame else VIEWER
    return replace(
        merged,
        frame=frame,
        subject_laterality_confirmed=laterality,
        notes=tuple(dict.fromkeys(base.notes + incoming.notes)),
    )
