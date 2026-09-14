"""Turn IMAGE_IR into a grounded prompt without adding anything to it.

The tiers exist so a prompt can be inspected at the altitude the mistake would
live at:

CORE    who or what the image shows, how they are posed, where they look.
DETAIL  the grounded expansion — wardrobe, scene, lighting, camera; the
        attributes that decide whether a generation still looks like the
        reference.
FINAL   the render-ready prompt — CORE and DETAIL, plus continuity wording that
        says the observed state holds, plus whatever micro-motion rule 6
        permits, plus the non-visual directives the author supplies.

The tiers were once called H1/H2/H3. They are not: "H3" is MiniMax H3, the
video model this project targets, and a field named h3_prompt that is not an H3
prompt is a trap for the next reader. The real H3 format lives in h3.py.

Composition is subtractive by construction. Every clause is built from a fact's
own recorded wording; the composer's only freedom is which facts to include,
how to order them, and which of a short fixed list of connectives to put
between sections. It never writes an adjective of its own, which is why its
output passes its own guard — an invariant the tests assert rather than assume.

Qualifier slots are the one piece of assembly it does. An attribute named
``<base>_<quality>`` — ``top_material``, ``hair_length`` — is folded into the
clause for ``<base>`` in the same section, so an uncertain material hedged as
"smooth" and a top observed as "blouse" compose into "smooth blouse": the
wording rule 7 asks for, reached without a special case for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .guard import AuditResult, audit
from .lexicon import EXCLUSIVE_GROUPS
from .motion import MotionPlan, MotionRejection, plan_motion
from .schema import ImageIR, SECTION_ORDER, tokenize

# Which subject attributes carry the core of the image, and in what order they
# read. Anything else in the subject section follows, alphabetically, in DETAIL.
CORE_SLOTS: tuple[str, ...] = ("identity", "subject", "description", "count", "pose", "posture", "stance", "gaze")

# Attributes whose value is a state that must be described as holding, not as
# arriving. These are exactly the attributes a motion prompt tends to animate.
CONTINUITY_SLOTS: tuple[str, ...] = (
    "gaze", "pose", "posture", "stance", "hands", "arms", "head", "expression", "orientation", "position",
)

# Qualities that read before the noun they qualify, in this order, the way a
# noun phrase normally stacks them.
QUALIFIER_ORDER: tuple[str, ...] = (
    "length", "size", "fit", "condition", "color", "colour", "tone", "pattern", "material", "fabric", "style",
)

# The only words the composer may put between sections that no fact supplies.
# Each is structural: it says how the next group of facts attaches, not what is
# in the picture. The guard's NEUTRAL_WORDS carries the same list.
SECTION_CONNECTIVES: dict[str, str] = {
    "subject": "",
    "wardrobe": "wearing",
    "scene": "in",
    "lighting": "lit by",
    "camera": "",
}


@dataclass(frozen=True)
class Clause:
    """One piece of prompt text and the IR fact(s) that license it."""

    text: str
    paths: tuple[str, ...]
    kind: str  # description | continuity | motion | directive


@dataclass
class Composition:
    core: str = ""
    detail: str = ""
    final: str = ""
    negative: str = ""
    clauses: list[Clause] = field(default_factory=list)
    motions: tuple[MotionPlan, ...] = ()
    rejected_motions: tuple[MotionRejection, ...] = ()
    unused_paths: tuple[str, ...] = ()
    self_audit: AuditResult | None = None

    def trace(self) -> str:
        """Clause-by-clause provenance: rule 9 made auditable before the fact."""
        lines = ["FINAL prompt trace", "=" * 18, ""]
        width = max((len(c.kind) for c in self.clauses), default=11)
        for clause in self.clauses:
            source = ", ".join(clause.paths) if clause.paths else "—"
            lines.append(f"[{clause.kind.ljust(width)}] {clause.text}\n{' ' * (width + 3)}<- {source}")
        if self.rejected_motions:
            lines.append("")
            lines.append("motion rejected")
            lines.append("-" * 15)
            for rejection in self.rejected_motions:
                lines.append(f"  {rejection.key}: {rejection.reason}")
        if self.unused_paths:
            lines.append("")
            lines.append("IMAGE_IR facts not used")
            lines.append("-" * 15)
            for path in self.unused_paths:
                lines.append(f"  {path}")
        return "\n".join(lines)


def _plural(slot: str) -> bool:
    return slot.endswith("s") and not slot.endswith(("ss", "us", "is"))


def _split_qualifiers(ir: ImageIR, section: str) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:
    """Map base slot -> [(quality, wording)], and the qualifier slots to skip.

    A qualifier only attaches when its base is itself emittable. When the base
    is missing the qualifier is skipped rather than emitted alone: "smooth" is
    not a description of anything, and "wearing smooth" would be a sentence the
    IR never licensed. It surfaces in the composition's unused facts instead, so
    the omission is visible rather than silent.
    """
    facts = {f.slot: f for f in ir.section(section)}
    attached: dict[str, list[tuple[str, str]]] = {}
    consumed: set[str] = set()
    for slot, fact in facts.items():
        text = fact.emitted_text()
        if not text or "_" not in slot:
            continue
        base, _, quality = slot.rpartition("_")
        base_fact = facts.get(base)
        if base_fact is None or not base_fact.emitted_text():
            if quality in QUALIFIER_ORDER:
                consumed.add(slot)
            continue
        attached.setdefault(base, []).append((quality, text))
        consumed.add(slot)
    for items in attached.values():
        items.sort(key=lambda item: (QUALIFIER_ORDER.index(item[0]) if item[0] in QUALIFIER_ORDER else len(QUALIFIER_ORDER), item[0]))
    return attached, consumed


def _label(section: str, slot: str, text: str, style: str) -> str:
    """Prefix a value with its attribute name where the value cannot stand alone.

    A subject attribute outside the core reads as a loose adjective on its own —
    "shoulder-length" says nothing until it says what is shoulder-length. Naming
    the attribute costs nothing in grounding, because a slot name is part of the
    IR. Sections that already carry a connective ("wearing ...", "in ...") are
    left bare: there the values are nouns and the connective frames them.
    """
    if style != "sentence" or section != "subject" or slot in CORE_SLOTS:
        return text
    label = slot.replace("_", " ")
    if set(tokenize(label)).issubset(set(tokenize(text))):
        return text
    return f"{label} {text}"


def _section_clauses(
    ir: ImageIR, section: str, slots: tuple[str, ...] | None = None, style: str = "sentence"
) -> list[Clause]:
    """Description clauses for one section, qualifiers folded into their base."""
    attached, consumed = _split_qualifiers(ir, section)
    facts = {f.slot: f for f in ir.section(section)}
    order = slots if slots is not None else tuple(sorted(facts))
    clauses: list[Clause] = []
    for slot in order:
        fact = facts.get(slot)
        if fact is None or slot in consumed:
            continue
        text = fact.emitted_text()
        if not text:
            continue
        paths = [fact.path]
        parts = []
        for quality, wording in attached.get(slot, []):
            parts.append(wording)
            paths.append(f"{section}.{slot}_{quality}")
        parts.append(text)
        clauses.append(
            Clause(text=_label(section, slot, " ".join(parts), style), paths=tuple(paths), kind="description")
        )
    return clauses


def _continuity_clauses(ir: ImageIR, used: set[str]) -> list[Clause]:
    """State the observed condition as holding — the legal alternative to rule 5.

    This is what replaces "she turns from looking away toward the camera": the
    same observed value, with the history removed and nothing put in its place.
    """
    clauses: list[Clause] = []
    for fact in ir.observed():
        if fact.slot not in CONTINUITY_SLOTS or not fact.value:
            continue
        label = fact.slot.replace("_", " ")
        verb = "stay" if _plural(fact.slot) else "stays"
        clauses.append(Clause(text=f"{label} {verb} {fact.value}", paths=(fact.path,), kind="continuity"))
        used.add(fact.path)
    return clauses


def _negative(ir: ImageIR) -> str:
    """Wordings the IR positively rules out, offered as a negative prompt.

    A negative prompt is an assertion: it says this is not in the picture. Only
    two things in an IR support that claim — an attribute recorded as absent
    (looked for, not there), and the unpinned poles of a group the IR pinned
    (gaze observed toward camera means it is not averted).

    Uncertain candidates are deliberately NOT here, and this is the correction
    that matters most. "material is uncertain, possibly satin or silk" means the
    reading could not be made; it does not mean the fabric is not satin. Putting
    those candidates in a negative prompt turns "we could not tell" into "it is
    definitely not that" — inventing an exclusion the image never supported, and
    steering generation away from what may well be the right answer. The
    candidates stay in the IR for the guard, which uses them to stop a prompt
    from *asserting* one; refusing to assert is not the same as denying.
    """
    observed = [(f.path, (f.value or "").lower()) for f in ir.observed() if f.value]
    out: list[str] = []
    for _group, poles in EXCLUSIVE_GROUPS:
        pinned = {
            index
            for index, pole in enumerate(poles)
            for phrase in pole
            if any(phrase in text for _path, text in observed)
        }
        if not pinned:
            continue
        for index, pole in enumerate(poles):
            if index not in pinned:
                out.append(pole[0])
    for fact in ir.facts:
        if fact.is_absent:
            out.append(fact.slot.replace("_", " "))
    return ", ".join(dict.fromkeys(term for term in out if term))


def compose(
    ir: ImageIR,
    *,
    motions: tuple[str, ...] | list[str] = (),
    require_anchor: bool = True,
    continuity: str = "auto",
    directives: str = "",
    style: str = "sentence",
) -> Composition:
    """Build CORE, DETAIL and FINAL from ``ir``, then audit the result against it.

    ``directives`` is the one input that is not traced to a fact: duration,
    aspect, codec, render style. It is kept verbatim in FINAL and marked as a
    directive in the trace, so a reader can see at a glance which part of the
    prompt is not answerable to the image.

    ``continuity`` adds "gaze stays toward camera" wording for the attributes a
    motion prompt tends to animate. "auto" adds it only when a micro-motion was
    accepted, which is when drift is actually a risk.
    """
    if style not in ("sentence", "tags"):
        raise ValueError(f"style must be 'sentence' or 'tags', got {style!r}")
    if continuity not in ("auto", "always", "never"):
        raise ValueError(f"continuity must be 'auto', 'always' or 'never', got {continuity!r}")

    used: set[str] = set()
    clauses: list[Clause] = []

    subject_slots = tuple(s for s in CORE_SLOTS if ir.get(f"subject.{s}"))
    core_clauses = _section_clauses(ir, "subject", subject_slots, style)
    clauses.extend(core_clauses)

    detail_groups: list[tuple[str, list[Clause]]] = []
    rest = tuple(sorted({f.slot for f in ir.section("subject")} - set(subject_slots)))
    subject_rest = _section_clauses(ir, "subject", rest, style)
    if subject_rest:
        detail_groups.append(("subject", subject_rest))
    for section in ir.sections():
        if section == "subject":
            continue
        group = _section_clauses(ir, section, None, style)
        if group:
            detail_groups.append((section, group))
    for _section, group in detail_groups:
        clauses.extend(group)

    for clause in clauses:
        used.update(clause.paths)

    def render(groups: list[tuple[str, list[Clause]]]) -> str:
        pieces: list[str] = []
        for section, group in groups:
            body = ", ".join(c.text for c in group)
            connective = SECTION_CONNECTIVES.get(section, "") if style == "sentence" else ""
            pieces.append(f"{connective} {body}".strip() if connective else body)
        return ", ".join(p for p in pieces if p)

    core = render([("subject", core_clauses)])
    detail = render(detail_groups)

    accepted, rejected = plan_motion(ir, tuple(motions), require_anchor=require_anchor)

    # "auto": continuity wording is what keeps a moving prompt from drifting off
    # the observed state, so it earns its length only once motion is in play.
    tail: list[Clause] = []
    if continuity == "always" or (continuity == "auto" and accepted):
        tail.extend(_continuity_clauses(ir, used))

    for plan in accepted:
        tail.append(Clause(text=plan.phrase, paths=(plan.trace,), kind="motion"))
        if plan.anchor:
            used.add(plan.anchor)

    directive_text = directives.strip().strip(",").strip()
    if directive_text:
        tail.append(Clause(text=directive_text, paths=(), kind="directive"))

    clauses.extend(tail)
    final_body = ", ".join(part for part in (core, detail, ", ".join(c.text for c in tail)) if part)
    final = (final_body.rstrip(", ") + ".") if final_body else ""

    composition = Composition(
        core=core,
        detail=detail,
        final=final,
        negative=_negative(ir),
        clauses=clauses,
        motions=accepted,
        rejected_motions=rejected,
        unused_paths=tuple(sorted(f.path for f in ir.facts if f.path not in used)),
    )
    composition.self_audit = audit(ir, final, require_anchor=require_anchor)
    return composition


__all__ = ["Clause", "Composition", "compose", "SECTION_ORDER", "tokenize"]
