"""A line-per-fact syntax for writing IMAGE_IR by hand.

JSON is the interchange format, but nobody transcribes what they see in an
image into nested braces. The IR is a flat list of attributes, and the point of
the syntax is that the certainty of each one is a single character — so marking
something uncertain is easier than leaving it observed by accident, which is
the direction the rules want the friction to run::

    @frame viewer
    @laterality unconfirmed

    subject.identity   = a woman
    subject.gaze       = toward camera
    subject.hands      = resting on the lap
    wardrobe.top       = blouse
    wardrobe.top_material ~ smooth | satin, silk
    subject.jewellery  !

``=`` observed · ``~`` uncertain, hedge before the pipe and the readings that
were weighed after it · ``!`` looked for and absent.

Trailing ``@`` markers annotate a line and may be combined::

    wardrobe.hosiery ~ sheer | nude tights @0.78 @verify

``@viewer`` / ``@subject`` override the document's geometry frame for that one
attribute; a bare number is the reader's confidence in its own reading; and
``@verify`` asks for a second look. Confidence is optional everywhere — a line
without it is making no claim about its own reliability, which is the honest
state for something a person typed.
"""

from __future__ import annotations

import re

from .schema import ABSENT, FRAMES, IRError, ImageIR, OBSERVED, UNCERTAIN, Fact

_SUFFIX = re.compile(r"\s+@([A-Za-z0-9_.]+)\s*$")
_PATH = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*\.[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$")


def _directive(line: str, number: int, state: dict) -> None:
    name, _, rest = line[1:].partition(" ")
    name = name.strip().lower()
    rest = rest.strip()
    if name == "frame":
        if rest not in FRAMES:
            raise IRError(f"line {number}: @frame must be one of {', '.join(FRAMES)}, got {rest!r}")
        state["frame"] = rest
    elif name == "laterality":
        if rest not in ("confirmed", "unconfirmed"):
            raise IRError(f"line {number}: @laterality must be 'confirmed' or 'unconfirmed', got {rest!r}")
        state["laterality"] = rest == "confirmed"
    elif name == "note":
        if rest:
            state["notes"].append(rest)
    else:
        raise IRError(f"line {number}: unknown directive @{name} (expected @frame, @laterality or @note)")


def _fact(line: str, number: int) -> Fact:
    frame: str | None = None
    confidence: float | None = None
    verification_required = False

    # Peeled right-to-left so the markers can be given in any order.
    while True:
        match = _SUFFIX.search(line)
        if not match:
            break
        marker = match.group(1)
        lowered = marker.lower()
        if lowered in FRAMES:
            frame = lowered
        elif lowered == "verify":
            verification_required = True
        else:
            try:
                confidence = float(marker)
            except ValueError:
                raise IRError(
                    f"line {number}: unknown marker @{marker} "
                    f"(expected @viewer, @subject, @verify or a confidence like @0.78)"
                ) from None
            if not 0.0 <= confidence <= 1.0:
                raise IRError(f"line {number}: confidence must be between 0.0 and 1.0, got {confidence}")
        line = line[: match.start()].rstrip()

    for marker, certainty in (("=", OBSERVED), ("~", UNCERTAIN)):
        head, sep, tail = line.partition(marker)
        if not sep:
            continue
        path = head.strip()
        if not _PATH.match(path):
            raise IRError(f"line {number}: {path!r} is not a section.attribute path")
        section, _, slot = path.partition(".")
        body = tail.strip()
        if certainty == OBSERVED:
            if not body:
                raise IRError(f"line {number}: {path} is marked observed but has no value (use '!' for absent)")
            return Fact(
                section=section, slot=slot, value=body, certainty=OBSERVED, frame=frame,
                confidence=confidence, verification_required=verification_required,
            )
        hedge, _, candidates = body.partition("|")
        return Fact(
            section=section,
            slot=slot,
            value=None,
            certainty=UNCERTAIN,
            hedge=hedge.strip() or None,
            candidates=tuple(c.strip() for c in candidates.split(",") if c.strip()),
            frame=frame,
            confidence=confidence,
            verification_required=verification_required,
        )

    if line.rstrip().endswith("!"):
        path = line.rstrip()[:-1].strip()
        if not _PATH.match(path):
            raise IRError(f"line {number}: {path!r} is not a section.attribute path")
        section, _, slot = path.partition(".")
        return Fact(
            section=section, slot=slot, value=None, certainty=ABSENT, frame=frame,
            confidence=confidence, verification_required=verification_required,
        )

    raise IRError(
        f"line {number}: expected 'section.attribute = value', 'section.attribute ~ hedge | candidates', "
        f"or 'section.attribute !'"
    )


def parse_dsl(text: str) -> ImageIR:
    """Parse the line syntax into an ``ImageIR``.

    A later line for the same attribute replaces an earlier one, so a block can
    be pasted over another without hunting for the line it supersedes.
    """
    state: dict = {"frame": "viewer", "laterality": False, "notes": []}
    facts: dict[str, Fact] = {}

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        if line.startswith("@"):
            _directive(line, number, state)
            continue
        fact = _fact(line, number)
        facts[fact.path] = fact

    return ImageIR(
        facts=tuple(facts.values()),
        frame=state["frame"],
        subject_laterality_confirmed=state["laterality"],
        notes=tuple(state["notes"]),
    )
