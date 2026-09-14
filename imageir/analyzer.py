"""Image in, IMAGE_IR out.

This is the stage that makes the rest of the project usable: without it every
graph starts with a human typing out what they can see, which is exactly the
labour the grounding rules were built to make safe rather than to require.

The delicate part is not the request — it is reading the reply. A vision model
asked for JSON returns clean JSON most of the time, and the rest of the time
returns it inside a markdown fence, after a paragraph of reasoning, before a
closing remark, or with a trailing comma. Those are all the same document
wearing different clothes, and refusing them wastes a real answer.

What is emphatically not allowed is repairing *content*. Stripping a fence or a
trailing comma changes nothing the model claimed; supplying a missing value,
picking one of two candidates, or promoting an uncertain reading to observed
would manufacture exactly the ungrounded facts this package exists to prevent.
So the repairs here are syntactic, exhaustively listed, and stop there: a reply
that is not recoverable by those alone is an error with an excerpt, not a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from .backend.base import BackendError, BackendResult, ImagePayload, VisionBackend, mask
from .schema import Fact, IRError, ImageIR, parse_ir

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "image_ir_extractor.txt"

DETAIL_LEVELS = ("fast", "balanced", "detailed")

# What each depth asks for beyond the system prompt. The instruction changes
# what is worth recording; the image resolution (see imaging.DETAIL_MAX_SIDE)
# changes what is legible at all.
DETAIL_INSTRUCTIONS = {
    "fast": (
        "Record the attributes that most decide whether a regeneration still looks like this image: "
        "subject, pose, gaze, main garments and their colours, scene, lighting and framing. "
        "Skip fine detail rather than guessing at it."
    ),
    "balanced": (
        "Work through the sections in the specification. Record what you can read and mark honestly "
        "what you cannot."
    ),
    "detailed": (
        "Work through every section and every risk area in the specification. Prefer more attributes "
        "carrying honest uncertainty over fewer attributes stated confidently. Give evidence for each "
        "reading in a risk area."
    ),
}

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


class AnalyzerError(RuntimeError):
    """The model's reply could not be read as IMAGE_IR. Already masked."""


def load_system_prompt(path: Path | None = None) -> str:
    """The extractor specification, from disk.

    A file rather than a Python constant: it is prose that gets tuned against
    real model behaviour, and prose wedged into a string literal stops being
    edited by the people best placed to improve it.
    """
    target = path or PROMPT_PATH
    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalyzerError(f"could not read the extractor prompt at {target}: {exc}") from None


def strip_fence(text: str) -> str:
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text


def find_json_object(text: str) -> str | None:
    """The first balanced ``{...}`` in ``text``, ignoring braces inside strings.

    Brace counting rather than a regular expression, because the document is
    nested and a regex cannot match balanced delimiters. String and escape
    awareness matters: a value like ``"a {curly} brace"`` would otherwise end
    the scan in the wrong place.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str, *, secrets: tuple = ()) -> dict:
    """Parse the JSON object out of a model reply.

    The repairs applied, in order, and the complete list of them:
      1. remove a surrounding markdown fence
      2. take the first balanced brace-delimited object, discarding prose
         before and after it
      3. remove trailing commas before a closing brace or bracket

    Nothing else. In particular nothing that supplies, alters or sharpens a
    value — a reply this cannot read is reported, not repaired into shape.
    """
    if not text or not text.strip():
        raise AnalyzerError("the model returned an empty reply")

    candidate = find_json_object(strip_fence(text))
    if candidate is None:
        raise AnalyzerError(
            mask(f"no JSON object found in the reply. It began: {text.strip()[:300]!r}", *secrets)
        )

    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            raise AnalyzerError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed

    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        excerpt = candidate[max(0, exc.pos - 120) : exc.pos + 120]
        raise AnalyzerError(
            mask(f"the reply is not valid JSON ({exc.msg} at position {exc.pos}) near: {excerpt!r}", *secrets)
        ) from None
    raise AnalyzerError("the reply could not be parsed")  # pragma: no cover - unreachable


@dataclass
class AnalysisResult:
    ir: ImageIR
    raw_reply: str = ""
    request_summary: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    flagged: tuple[Fact, ...] = ()

    def low_confidence_report(self, threshold: float) -> str:
        """The attributes a second reader should look at, and why."""
        if not self.flagged:
            return f"no attribute fell below confidence {threshold:g} or asked for verification."
        lines = [f"attributes below confidence {threshold:g}, or flagged by the reader:", ""]
        for fact in self.flagged:
            score = "unstated" if fact.confidence is None else f"{fact.confidence:.2f}"
            lines.append(f"  {fact.path}  [{fact.certainty}]  confidence {score}")
            if fact.candidates:
                lines.append(f"      weighed: {', '.join(fact.candidates)}")
            if fact.evidence:
                lines.append(f"      basis:   {fact.evidence}")
        return "\n".join(lines)

    def compact_summary(self) -> str:
        """One line per attribute, in the IR's own wording."""
        lines = []
        for section in self.ir.sections():
            readings = []
            for fact in self.ir.section(section):
                if fact.is_observed:
                    readings.append(f"{fact.slot}={fact.value}")
                elif fact.is_uncertain:
                    readings.append(f"{fact.slot}~{fact.hedge or '?'}")
                else:
                    readings.append(f"{fact.slot}!")
            if readings:
                lines.append(f"{section}: " + ", ".join(readings))
        return "\n".join(lines)


def flag_low_confidence(ir: ImageIR, threshold: float) -> tuple[ImageIR, tuple[Fact, ...]]:
    """Mark readings below ``threshold`` for verification, and return them.

    The marking is additive and says only what already happened: this reading
    scored below the threshold the user set. It never changes a value, a hedge
    or a certainty — the routing that would act on the flag is deliberately not
    built yet, and the flag has to survive until it is.
    """
    flagged: list[Fact] = []
    updated = ir
    for fact in ir.facts:
        below = fact.confidence is not None and fact.confidence < threshold
        if below or fact.verification_required:
            marked = fact if fact.verification_required else replace(fact, verification_required=True)
            updated = updated.with_fact(marked)
            flagged.append(marked)
    return updated, tuple(flagged)


def build_user_prompt(detail: str) -> str:
    if detail not in DETAIL_LEVELS:
        raise AnalyzerError(f"analysis_detail must be one of {', '.join(DETAIL_LEVELS)}, got {detail!r}")
    return (
        "Analyse this image and return IMAGE_IR as specified.\n\n"
        f"{DETAIL_INSTRUCTIONS[detail]}\n\n"
        "Return the JSON object only."
    )


def analyze(
    backend: VisionBackend,
    image: ImagePayload,
    *,
    detail: str = "balanced",
    confidence_threshold: float = 0.6,
    system_prompt: str | None = None,
) -> AnalysisResult:
    """Describe ``image`` through ``backend`` and return validated IMAGE_IR."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise AnalyzerError(f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}")

    instructions = (system_prompt or "").strip() or load_system_prompt()
    secrets = (backend.config.api_token,)

    try:
        result: BackendResult = backend.analyze_image(
            image=image,
            system_prompt=instructions,
            user_prompt=build_user_prompt(detail),
        )
    except BackendError as exc:
        raise AnalyzerError(str(exc)) from None

    data = extract_json(result.text, secrets=secrets)
    try:
        ir = parse_ir(data)
    except IRError as exc:
        # The model produced JSON but not IMAGE_IR. Naming the offending path
        # is what makes this fixable by editing the extractor prompt.
        raise AnalyzerError(mask(f"the reply is not valid IMAGE_IR: {exc}", *secrets)) from None

    if not ir.facts:
        raise AnalyzerError("the reply parsed but recorded no attributes at all")

    ir, flagged = flag_low_confidence(ir, confidence_threshold)
    return AnalysisResult(
        ir=ir,
        raw_reply=mask(result.text, *secrets),
        request_summary=result.request_summary,
        model=result.model,
        usage=result.usage,
        flagged=flagged,
    )
