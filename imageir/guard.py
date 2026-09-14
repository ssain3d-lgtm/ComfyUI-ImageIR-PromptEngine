"""Rule 9 — check every visual statement in a prompt against IMAGE_IR.

The guard is the reason the rest of the package can be trusted. Composition can
be careful; a human edit afterwards, a style suffix pasted from somewhere else,
or a prompt written entirely by hand cannot be assumed to be. So the last stage
re-derives the verdict from the text itself: it splits the prompt into clauses,
classifies each one, and keeps only what it can trace back to an IR fact, to the
rule 6 allowlist, or to a directive that makes no claim about the image at all.

A clause is judged in a fixed order, strictest first, because the findings are
not interchangeable. "She begins to blink" contains a legal micro-motion and an
illegal invented beginning; reported as a micro-motion it would read as clean.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lexicon import (
    DISALLOWED_MOTION_PATTERNS,
    EQUIVALENCES,
    EXCLUSIVE_GROUPS,
    MICRO_MOTION_PATTERNS,
    MICRO_MOTIONS,
    SLOT_KIND_SPECIFICS,
    STYLE_TERMS,
    SUBJECT_LATERALITY_PATTERNS,
    TECHNICAL_TERMS,
    TRANSITION_PATTERNS,
    VIEWER_LATERALITY_HINT,
)
from .motion import plan_motion
from .schema import ImageIR, tokenize

# The numbered rules, quoted so a report can name the line it enforced rather
# than only its number.
RULE_TEXT: dict[int, str] = {
    1: "Never introduce a visual fact that is not present in IMAGE_IR.",
    2: "Never change an observed attribute.",
    3: "Never resolve an 'uncertain' attribute into a specific value.",
    4: "Never convert viewer-left / viewer-right into subject-left / subject-right "
       "unless subject laterality is explicitly confirmed.",
    5: "Do not invent an initial state for motion.",
    6: "Motion may be newly generated ONLY when it does not contradict IMAGE_IR.",
    7: "Preserve uncertainty.",
    8: "When IMAGE_IR provides viewer-relative geometry, preserve viewer-relative terminology.",
    9: "Compare every visual statement against IMAGE_IR; remove what cannot be traced.",
}

# Clause boundaries. Splitting on commas as well as sentence marks is what makes
# a surgical fix possible: one invented detail can be lifted out of a sentence
# without taking the grounded half of it along.
_SPLIT_RE = re.compile(r"([.;!?\n]+|,)")

# Structural words that carry no attribute of their own: they say that
# something continues, or name the picture rather than anything in it. Kept out
# of the "untraceable" count so a continuity clause is not read as a new claim.
NEUTRAL_WORDS: frozenset[str] = frozenset(
    """
    subject figure person people frame image picture photograph shot scene composition
    wearing lit against framed
    remains remain remaining stays stay staying holds hold holding keeps keep keeping
    continues continue continuing maintains maintain maintaining
    unchanged same throughout entire whole steady constant no not none never without nothing
    very slight slightly barely faint faintest subtle small tiny minute minimal gentle gently
    quiet quietly natural naturally unhurried few little more less than other another
    visible seen shown appears appear present
    """.split()
)

# Words the rule 6 phrasings need. A permitted micro-motion clause is kept whole
# before this matters, but a clause that mixes one with description still has to
# resolve, and these words are licensed by the allowlist rather than by the IR.
MOTION_VOCAB: frozenset[str] = frozenset(
    word for motion in MICRO_MOTIONS for word in tokenize(str(motion["phrase"]))
)

# Which pronouns a prompt may use for the subject. "they" asserts nothing and is
# always available; a gendered pronoun asserts how the subject presents, so it
# needs the IR to have recorded that.
GENDERED_PRONOUNS: dict[str, tuple[str, ...]] = {
    "she": ("woman", "women", "female", "girl", "lady", "she", "her", "feminine"),
    "her": ("woman", "women", "female", "girl", "lady", "she", "her", "feminine"),
    "hers": ("woman", "women", "female", "girl", "lady", "she", "her", "feminine"),
    "he": ("man", "men", "male", "boy", "gentleman", "he", "his", "masculine"),
    "his": ("man", "men", "male", "boy", "gentleman", "he", "his", "masculine"),
    "him": ("man", "men", "male", "boy", "gentleman", "he", "his", "masculine"),
}


def equivalence_index(ir: ImageIR) -> dict[str, tuple[str, ...]]:
    """Word -> IR paths, for paraphrases of wordings this IR observed.

    Only observed facts open a paraphrase: an uncertain attribute has no
    wording to be equivalent to, and admitting one would resolve it.
    """
    observed = [(f.path, (f.value or "").lower()) for f in ir.observed() if f.value]
    index: dict[str, set[str]] = {}
    for canonical, paraphrases in EQUIVALENCES:
        paths = {path for path, text in observed if canonical in text}
        if not paths:
            continue
        for phrase in paraphrases:
            for word in tokenize(phrase):
                index.setdefault(word, set()).update(paths)
    return {word: tuple(sorted(paths)) for word, paths in index.items()}


def absence_index(ir: ImageIR) -> dict[str, tuple[str, ...]]:
    """Word -> IR paths, for attributes the IR recorded as absent.

    An absent fact licenses no wording of its own, so it is not in the IR's
    vocabulary. It does license *naming* the attribute in order to deny it —
    "no jewellery" — which is how an absence reaches a negative prompt. The
    affirmative form is rejected before tracing ever runs, so admitting these
    words here cannot let a described absence through.
    """
    index: dict[str, set[str]] = {}
    for fact in ir.facts:
        if not fact.is_absent:
            continue
        for word in tokenize(fact.slot.replace("_", " ")):
            index.setdefault(word, set()).add(fact.path)
    return {word: tuple(sorted(paths)) for word, paths in index.items()}


def allowed_pronouns(ir: ImageIR) -> frozenset[str]:
    """Pronouns this IR licenses. Neutral ones always; a gendered one only when
    IMAGE_IR recorded how the subject presents."""
    vocabulary = ir.vocabulary()
    return frozenset(
        word for word, grounds in GENDERED_PRONOUNS.items() if vocabulary.intersection(grounds)
    )


@dataclass(frozen=True)
class Finding:
    """One rule broken by one clause, with the fix that would satisfy it."""

    rule: int
    label: str
    clause: str
    detail: str
    suggestion: str | None = None

    def render(self) -> str:
        line = f"rule {self.rule} · {self.label}: {self.detail}\n    in: {self.clause.strip()}"
        if self.suggestion:
            line += f"\n    fix: {self.suggestion}"
        return line


@dataclass(frozen=True)
class ClauseVerdict:
    text: str
    separator: str
    classification: str  # grounded | micro_motion | technical | structural | violation
    trace: tuple[str, ...] = ()
    findings: tuple[Finding, ...] = ()

    @property
    def kept(self) -> bool:
        return self.classification != "violation"


@dataclass(frozen=True)
class AuditResult:
    clauses: tuple[ClauseVerdict, ...]
    filtered_prompt: str
    ir_paths_used: tuple[str, ...]

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(f for c in self.clauses for f in c.findings)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def removed(self) -> tuple[str, ...]:
        return tuple(c.text.strip() for c in self.clauses if not c.kept)

    def report(self) -> str:
        """A human-readable audit, listing what was kept and why, then the breaches."""
        lines = ["IMAGE_IR grounding audit", "=" * 24, ""]
        if self.clean:
            lines.append("verdict: clean — every visual statement traces to IMAGE_IR.")
        else:
            lines.append(f"verdict: {len(self.findings)} violation(s); {len(self.removed)} clause(s) removed.")
        lines.append("")
        lines.append("clauses")
        lines.append("-" * 24)
        for clause in self.clauses:
            text = clause.text.strip()
            if not text:
                continue
            mark = "kept  " if clause.kept else "REMOVE"
            trace = f"  <- {', '.join(clause.trace)}" if clause.trace else ""
            lines.append(f"  [{mark}] ({clause.classification}) {text}{trace}")
        if self.findings:
            lines.append("")
            lines.append("violations")
            lines.append("-" * 24)
            for finding in self.findings:
                lines.append("  " + finding.render().replace("\n", "\n  "))
                lines.append(f"    rule {finding.rule}: {RULE_TEXT[finding.rule]}")
        return "\n".join(lines)


# Private-use codepoints. A protected span is swapped for one of these before
# auditing: they match no token pattern, so every check sees the span as absent
# rather than as a word it has to judge.
_SHIELD_BASE = 0xE000
_SHIELD_LIMIT = 0x900


def audit_protected(
    ir: ImageIR,
    text: str,
    protected: tuple,
    **kwargs,
) -> tuple[AuditResult, str]:
    """Audit ``text`` while treating matches of ``protected`` as untouchable.

    Structured prompt formats carry syntax that is not a claim about the image:
    field labels, shot markers, picture references, speaker ids. Passing that
    syntax through the clause classifier produces two failures at once — the
    markers are flagged as untraceable, and removing them destroys the format.

    Shielding is the honest fix, and it is stronger than the obvious
    alternative of adding the markers to an allowlist. An allowlisted word is
    still a word in the sentence, so "<Picture 1> wearing a satin blouse" would
    have its marker forgiven and could drag the rest of the clause along.
    A shielded span is *removed* before judging and restored after, so the
    grounding rules run on exactly the prose that makes visual claims, with the
    same strictness they always had.

    Returns the audit of the shielded text, and the filtered text with the
    markers put back.
    """
    spans: list[str] = []

    def shield(match) -> str:
        if len(spans) >= _SHIELD_LIMIT:
            return match.group(0)
        spans.append(match.group(0))
        return chr(_SHIELD_BASE + len(spans) - 1)

    shielded = text
    for pattern in protected:
        shielded = pattern.sub(shield, shielded)

    result = audit(ir, shielded, **kwargs)

    def restore(value: str) -> str:
        for index, original in enumerate(spans):
            value = value.replace(chr(_SHIELD_BASE + index), original)
        return value

    restored_clauses = tuple(
        ClauseVerdict(
            text=restore(clause.text),
            separator=clause.separator,
            classification=clause.classification,
            trace=clause.trace,
            findings=tuple(
                Finding(
                    rule=f.rule,
                    label=f.label,
                    clause=restore(f.clause),
                    detail=f.detail,
                    suggestion=f.suggestion,
                )
                for f in clause.findings
            ),
        )
        for clause in result.clauses
    )
    readable = AuditResult(
        clauses=restored_clauses,
        filtered_prompt=restore(result.filtered_prompt),
        ir_paths_used=result.ir_paths_used,
    )
    return readable, readable.filtered_prompt


def _split(prompt: str) -> list[tuple[str, str]]:
    """Prompt -> [(clause text, the punctuation that followed it)]."""
    parts = _SPLIT_RE.split(prompt)
    out: list[tuple[str, str]] = []
    for index in range(0, len(parts), 2):
        text = parts[index]
        separator = parts[index + 1] if index + 1 < len(parts) else ""
        if text.strip() or separator.strip():
            out.append((text, separator))
    return out


def _observed_texts(ir: ImageIR) -> list[tuple[str, str]]:
    return [(f.path, (f.value or "").lower()) for f in ir.observed() if f.value]


def _check_contradiction(ir: ImageIR, clause: str) -> Finding | None:
    """Rule 2 — the clause asserts a pole the IR observed to be a different one."""
    low = clause.lower()
    observed = _observed_texts(ir)
    for group_name, poles in EXCLUSIVE_GROUPS:
        ir_poles: dict[int, tuple[str, str]] = {}
        for index, pole in enumerate(poles):
            for phrase in pole:
                for path, text in observed:
                    if phrase in text:
                        ir_poles.setdefault(index, (path, phrase))
        if not ir_poles:
            continue
        for index, pole in enumerate(poles):
            if index in ir_poles:
                continue
            for phrase in pole:
                if re.search(rf"\b{re.escape(phrase)}\b", low):
                    path, observed_phrase = next(iter(ir_poles.values()))
                    return Finding(
                        rule=2,
                        label="changed observed attribute",
                        clause=clause,
                        detail=(
                            f"{group_name}: the prompt says {phrase!r} but {path} observed "
                            f"{observed_phrase!r}"
                        ),
                        suggestion=f"restate {path} as observed: {observed_phrase!r}",
                    )
    return None


def _check_uncertainty(ir: ImageIR, clause: str) -> Finding | None:
    """Rules 3 and 7 — the clause picks a specific value the image did not give."""
    low = clause.lower()
    for fact in ir.facts:
        if not fact.is_uncertain:
            continue
        for specific in fact.forbidden_specifics():
            if re.search(rf"\b{re.escape(specific.lower())}\b", low):
                return Finding(
                    rule=3,
                    label="resolved an uncertain attribute",
                    clause=clause,
                    detail=f"{fact.path} is uncertain, but the prompt commits to {specific!r}",
                    suggestion=(
                        f"use the hedge {fact.hedge!r}" if fact.hedge else f"drop the wording for {fact.path}"
                    ),
                )
        kind = next((k for k in SLOT_KIND_SPECIFICS if k in fact.slot.lower()), None)
        if kind is None:
            continue
        hedge_words = set(tokenize(fact.hedge))
        for specific in SLOT_KIND_SPECIFICS[kind]:
            if specific in hedge_words:
                continue
            if re.search(rf"\b{re.escape(specific)}\b", low):
                return Finding(
                    rule=7,
                    label="resolved an uncertain attribute",
                    clause=clause,
                    detail=f"{fact.path} ({kind}) is uncertain, but the prompt names {specific!r}",
                    suggestion=(
                        f"use the hedge {fact.hedge!r}" if fact.hedge else f"drop the {kind} wording"
                    ),
                )
    return None


# Saying an attribute is *not* there agrees with the IR rather than
# contradicting it, so a negated mention is the one way an absent attribute may
# be named — which is also how it reaches a negative prompt.
_NEGATIONS = frozenset({"no", "not", "none", "never", "without", "nothing", "absent", "bare"})


def _check_absent(ir: ImageIR, clause: str) -> Finding | None:
    """Rule 1 — the clause describes something the IR looked for and did not find."""
    low = clause.lower()
    if _NEGATIONS.intersection(tokenize(clause)):
        return None
    for fact in ir.facts:
        if not fact.is_absent:
            continue
        words = tokenize(fact.slot.replace("_", " "))
        if words and all(re.search(rf"\b{re.escape(word)}\b", low) for word in words):
            return Finding(
                rule=1,
                label="described an absent attribute",
                clause=clause,
                detail=f"{fact.path} is recorded as absent in IMAGE_IR",
                suggestion="remove the clause",
            )
    return None


def _check_laterality(ir: ImageIR, clause: str) -> Finding | None:
    """Rules 4 and 8 — subject-relative sides without confirmed laterality."""
    if ir.subject_laterality_confirmed:
        return None
    for pattern in SUBJECT_LATERALITY_PATTERNS:
        match = pattern.search(clause)
        if match:
            return Finding(
                rule=4,
                label="subject-relative laterality",
                clause=clause,
                detail=(
                    f"{match.group(0)!r} is subject-relative, but IMAGE_IR geometry is "
                    f"{ir.frame}-relative and subject laterality is not confirmed"
                ),
                suggestion=f"say {VIEWER_LATERALITY_HINT}",
            )
    return None


def _check_transition(clause: str) -> Finding | None:
    """Rule 5 — the clause places the observed state at the end of a change."""
    for label, pattern in TRANSITION_PATTERNS:
        match = pattern.search(clause)
        if match:
            return Finding(
                rule=5,
                label="invented initial state",
                clause=clause,
                detail=f"{match.group(0)!r} implies {label}; a still frame has no state before itself",
                suggestion="state the observed condition as continuing, e.g. 'remains ...'",
            )
    return None


def _match_micro_motion(clause: str) -> str | None:
    for key, pattern in MICRO_MOTION_PATTERNS:
        if pattern.search(clause):
            return key
    return None


def _check_disallowed_motion(clause: str, traceable: bool) -> Finding | None:
    """Rule 6 — motion outside the allowlist.

    Skipped for a clause the IR already accounts for: "hands resting on the lap"
    is the pose the image recorded, not a hand being placed there.
    """
    if traceable:
        return None
    for label, pattern in DISALLOWED_MOTION_PATTERNS:
        match = pattern.search(clause)
        if match:
            return Finding(
                rule=6,
                label="unsupported motion",
                clause=clause,
                detail=f"{match.group(0)!r} is {label}, which a still frame cannot support",
                suggestion="remove it, or keep only an allowed micro-motion",
            )
    return None


def _untraceable_words(
    ir: ImageIR, clause: str, allow_style: bool, equivalents: frozenset[str], check_pronouns: bool
) -> tuple[str, ...]:
    """Content words in the clause that nothing licenses.

    Bare numerals are exempt: on their own they are parameters — a duration, an
    aspect ratio, a focal length — not a claim about the picture. Counting words
    ("two people") are spelled out and stay subject to the same check as any
    other description.
    """
    vocabulary = ir.vocabulary()
    pronouns = allowed_pronouns(ir) if check_pronouns else frozenset(GENDERED_PRONOUNS)
    allowed = (
        vocabulary | NEUTRAL_WORDS | MOTION_VOCAB | TECHNICAL_TERMS | pronouns | equivalents
        | frozenset(absence_index(ir))
    )
    if allow_style:
        allowed = allowed | STYLE_TERMS
    return tuple(word for word in tokenize(clause) if word not in allowed and not word.isdigit())


def _check_pronoun(ir: ImageIR, clause: str) -> Finding | None:
    allowed = allowed_pronouns(ir)
    for word in tokenize(clause):
        if word in GENDERED_PRONOUNS and word not in allowed:
            return Finding(
                rule=1,
                label="ungrounded pronoun",
                clause=clause,
                detail=f"{word!r} asserts how the subject presents, which IMAGE_IR does not record",
                suggestion="say 'the subject' or 'they'",
            )
    return None


def audit(
    ir: ImageIR,
    prompt: str,
    *,
    allow_style: bool = True,
    require_anchor: bool = True,
    check_pronouns: bool = True,
) -> AuditResult:
    """Classify every clause of ``prompt`` against ``ir`` and drop what fails.

    ``allow_style`` lets render-style wording ("cinematic", "film grain")
    through: it changes how the picture looks without claiming anything about
    what is in it. Turn it off for a prompt that must describe the image and nothing else.
    """
    verdicts: list[ClauseVerdict] = []
    used: set[str] = set()
    index = ir.trace_index()
    for extra in (equivalence_index(ir), absence_index(ir)):
        for word, paths in extra.items():
            index[word] = tuple(sorted(set(index.get(word, ())) | set(paths)))
    equivalents = frozenset(equivalence_index(ir))

    for text, separator in _split(prompt):
        words = tokenize(text)
        if not words:
            verdicts.append(ClauseVerdict(text=text, separator=separator, classification="structural"))
            continue

        # Order matters: a transition is a breach even inside a legal motion.
        finding = _check_transition(text)
        if finding is None:
            finding = _check_laterality(ir, text)
        if finding is None:
            finding = _check_absent(ir, text)
        if finding is None:
            finding = _check_uncertainty(ir, text)
        if finding is None:
            finding = _check_contradiction(ir, text)
        if finding is None and check_pronouns:
            finding = _check_pronoun(ir, text)

        if finding is not None:
            verdicts.append(
                ClauseVerdict(text=text, separator=separator, classification="violation", findings=(finding,))
            )
            continue

        unmatched = _untraceable_words(ir, text, allow_style, equivalents, check_pronouns)
        traceable = not unmatched
        trace = tuple(sorted({path for word in words for path in index.get(word, ())}))

        motion_key = _match_micro_motion(text)
        if motion_key is not None:
            accepted, rejected = plan_motion(ir, [motion_key], require_anchor=require_anchor)
            if rejected:
                verdicts.append(
                    ClauseVerdict(
                        text=text,
                        separator=separator,
                        classification="violation",
                        findings=(
                            Finding(
                                rule=6,
                                label="unsupported micro-motion",
                                clause=text,
                                detail=f"{motion_key}: {rejected[0].reason}",
                                suggestion="remove it, or record the attribute in IMAGE_IR first",
                            ),
                        ),
                    )
                )
                continue
            used.update(plan.anchor for plan in accepted if plan.anchor)
            verdicts.append(
                ClauseVerdict(
                    text=text,
                    separator=separator,
                    classification="micro_motion",
                    trace=trace or (f"rule 6 · {motion_key}",),
                )
            )
            continue

        finding = _check_disallowed_motion(text, traceable)
        if finding is not None:
            verdicts.append(
                ClauseVerdict(text=text, separator=separator, classification="violation", findings=(finding,))
            )
            continue

        if unmatched:
            verdicts.append(
                ClauseVerdict(
                    text=text,
                    separator=separator,
                    classification="violation",
                    findings=(
                        Finding(
                            rule=1,
                            label="untraceable visual fact",
                            clause=text,
                            detail="no IMAGE_IR fact licenses: " + ", ".join(repr(w) for w in unmatched),
                            suggestion="remove it, or add the observation to IMAGE_IR",
                        ),
                    ),
                )
            )
            continue

        used.update(trace)
        if trace:
            classification = "grounded"
        elif set(words) & (TECHNICAL_TERMS | STYLE_TERMS):
            classification = "technical"
        else:
            classification = "structural"
        verdicts.append(
            ClauseVerdict(text=text, separator=separator, classification=classification, trace=trace)
        )

    return AuditResult(
        clauses=tuple(verdicts),
        filtered_prompt=rejoin(verdicts),
        ir_paths_used=tuple(sorted(used)),
    )


def rejoin(verdicts: list[ClauseVerdict] | tuple[ClauseVerdict, ...]) -> str:
    """Rebuild a prompt from the kept clauses, without leaving stray punctuation."""
    pieces: list[str] = []
    for verdict in verdicts:
        if not verdict.kept:
            continue
        text = verdict.text.strip()
        if not text:
            continue
        separator = verdict.separator.strip()
        pieces.append(text + (separator if separator in (",", ".", ";", "!", "?") else ","))
    if not pieces:
        return ""
    joined = " ".join(pieces).strip()
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r"\s+([,.;!?])", r"\1", joined)
    joined = re.sub(r"[,;]\s*([,.;!?])", r"\1", joined)
    return joined.rstrip(",; ").rstrip() + ("" if joined.endswith((".", "!", "?")) else ".")
