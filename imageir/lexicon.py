"""Word tables the composer and the guard share.

Everything here is deliberately lexical and deterministic. A model-based check
would be more supple, but a grounding rule that only *usually* fires is not a
grounding rule — a reviewer cannot tell the difference between "this prompt is
clean" and "the checker had an off day". Every table below is data a user can
read, extend and argue with, and every decision the guard makes points back at
a named entry in one of them.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# Rule 6 — the micro-motions a prompt may add on its own.
#
# A still frame licenses only motion that leaves every visible attribute where
# the image put it. These five do: the subject is doing them in the frame
# already, at some point in their cycle, whether or not the frame caught it.
# `conflicts` names IR wordings that would make even these contradict the
# image — a blink needs eyes that are open in the first place.
# --------------------------------------------------------------------------
MICRO_MOTIONS: tuple[dict[str, object], ...] = (
    {
        "key": "blink",
        "phrase": "a natural unhurried blink",
        "slots": ("subject.eyes", "subject.gaze", "subject.face"),
        "conflicts": ("eyes closed", "closed eyes", "eyes shut", "blindfold"),
    },
    {
        "key": "breathing",
        "phrase": "quiet breathing that barely moves the shoulders",
        "slots": ("subject.pose", "subject.posture", "subject.torso"),
        "conflicts": (),
    },
    {
        "key": "finger_movement",
        "phrase": "the faintest movement in the fingers",
        "slots": ("subject.hands", "subject.fingers"),
        "conflicts": ("hands out of frame", "hands not visible", "hands hidden", "gloved"),
    },
    {
        "key": "hair_movement",
        "phrase": "a few strands of hair drifting",
        "slots": ("subject.hair",),
        "conflicts": ("hair tied back tightly", "hair covered", "headscarf", "bald", "shaved head"),
    },
    {
        "key": "weight_shift",
        "phrase": "a barely perceptible settling of weight",
        "slots": ("subject.pose", "subject.posture", "subject.stance"),
        "conflicts": ("lying down", "reclining", "seated on the floor"),
    },
)

MICRO_MOTION_KEYS: tuple[str, ...] = tuple(str(m["key"]) for m in MICRO_MOTIONS)

# Wordings that read as one of the five above. A prompt is allowed to phrase a
# micro-motion its own way, so the guard recognises the motion rather than the
# exact sentence the composer would have written.
MICRO_MOTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (key, re.compile(pattern, re.IGNORECASE))
    for key, pattern in (
        ("blink", r"\bblink(?:s|ing)?\b"),
        ("breathing", r"\b(?:breath(?:e|es|ing)?|inhal\w+|exhal\w+|respiration)\b"),
        ("finger_movement", r"\b(?:finger|fingers|fingertip\w*)\b.{0,40}?\b(?:move\w*|twitch\w*|flex\w*|curl\w*|stir\w*|shift\w*)\b"),
        ("finger_movement", r"\b(?:move\w*|twitch\w*|flex\w*|curl\w*|stir\w*)\b.{0,20}?\b(?:finger|fingers)\b"),
        ("hair_movement", r"\b(?:hair|strand\w*|lock\w*)\b.{0,40}?\b(?:drift\w*|stir\w*|sway\w*|move\w*|float\w*|shift\w*|flutter\w*)\b"),
        ("hair_movement", r"\b(?:drift\w*|stir\w*|sway\w*|flutter\w*)\b.{0,20}?\b(?:hair|strand\w*)\b"),
        ("weight_shift", r"\b(?:weight|balance|posture)\b.{0,30}?\b(?:shift\w*|settl\w*|adjust\w*|sway\w*)\b"),
        ("weight_shift", r"\b(?:shift\w*|settl\w*|adjust\w*)\b.{0,30}?\b(?:weight|balance)\b"),
    )
)

# --------------------------------------------------------------------------
# Rule 5 — no invented initial state.
#
# A still frame has no "before". Any wording that puts the observed state at
# the end of a change has invented the state it changed from, even when the
# observed state itself is quoted correctly: "she turns from looking away
# toward the camera" gets the gaze right and the history wrong.
# --------------------------------------------------------------------------
TRANSITION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("a state before the frame", r"\b(?:at first|initially|to begin with|at the start|in the beginning|to start)\b"),
        ("a state before the frame", r"\b(?:previously|used to|had been|having been|was (?:looking|facing|standing|seated|holding))\b"),
        ("a change of state", r"\b(?:begin|begins|beginning|start|starts|starting)\s+(?:to|by)\b"),
        ("a change of state", r"\bturn(?:s|ing)?\s+from\b"),
        ("a change of state", r"\bfrom\s+\w+(?:\s+\w+){0,3}\s+(?:to|into|toward|towards)\s+\w+"),
        ("a change of state", r"\b(?:then|after that|afterwards?|next|finally|eventually|by the end|once more)\b"),
        ("a change of state", r"\b(?:goes|shifts|changes|transitions|moves)\s+from\b"),
        ("a change of state", r"\b(?:gradually|slowly|suddenly)\s+(?:turn\w*|rais\w*|lower\w*|open\w*|clos\w*|look\w*|shift\w*|lean\w*)\b"),
        ("a change of state", r"\bbefore\s+(?:she|he|they|it|the subject)\b"),
    )
)

# --------------------------------------------------------------------------
# Rule 6 — motion the image cannot support.
#
# Each entry names the category from the rule, so the report tells a user which
# line of the contract the prompt crossed rather than only that it crossed one.
# These are checked after IR tracing and after the micro-motion table: a pose
# the IR actually observed ("hands resting on the lap") is a description, not a
# movement, and must not be flagged for containing "rest".
# --------------------------------------------------------------------------
DISALLOWED_MOTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        (
            "changing gaze direction",
            r"\b(?:turn\w*|shift\w*|avert\w*|redirect\w*|lift\w*|drop\w*|move\w*)\s+(?:her|his|their|the)?\s*"
            r"(?:gaze|eyes|eyeline|head|look)\b",
        ),
        ("changing gaze direction", r"\b(?:looks?|looking|glanc\w+|peer\w+|gaz\w+)\s+(?:away|up|down|back|over|off|aside|elsewhere)\b"),
        ("changing gaze direction", r"\b(?:meets?|finds?|catches?)\s+(?:the\s+)?(?:camera|lens|viewer)\b"),
        ("changing gaze direction", r"\b(?:closes?|opens?)\s+(?:her|his|their)\s+eyes\b"),
        (
            "changing pose category",
            r"\b(?:stand(?:s|ing)?\s+up|sit(?:s|ting)?\s+down|kneel\w*|lie\w*\s+down|ris(?:es|ing)|get(?:s|ting)?\s+up"
            r"|crouch\w*|squat\w*|turn\w*\s+around|spin\w*|twist\w*|pivot\w*|bend\w*\s+(?:over|down|forward))\b",
        ),
        ("changing pose category", r"\blean(?:s|ing)?\s+(?:in|back|forward|closer|against|over)\b"),
        (
            "raising or lowering a limb",
            r"\b(?:rais\w+|lift\w+|lower\w+|drop\w+|extend\w+|stretch\w+|fold\w+|cross\w+|uncross\w+|wav\w+|point\w+|swing\w+)"
            r"\s+(?:her|his|their|the|a|an|one|both)?\s*"
            r"(?:arm|arms|hand|hands|leg|legs|finger|fingers|elbow|elbows|shoulder|shoulders|chin|head|knee|knees|foot|feet)\b",
        ),
        (
            "touching a new body part",
            r"\b(?:touch\w*|brush\w*|strok\w*|caress\w*|scratch\w*|rub\w*|tuck\w*|cup\w*|clasp\w*|grip\w*)\s+"
            r"(?:her|his|their|the|a|an)?\s*"
            r"(?:hair|face|cheek|chin|neck|arm|shoulder|lip|lips|ear|forehead|hand|wrist|knee|leg)\b",
        ),
        (
            "touching a new body part",
            r"\b(?:place\w*|rest\w*|put\w*|lay\w*|slide\w*|run\w*)\s+(?:her|his|their)\s+"
            r"(?:hand|hands|fingers|palm|arm|arms)\s+(?:on|onto|over|against|through|across|behind|under)\b",
        ),
        (
            "moving to another location",
            r"\b(?:walk\w*|step\w*|strid\w*|approach\w*|enter\w*|exit\w*|leav\w*|depart\w*|wander\w*|pac\w+)\b",
        ),
        ("moving to another location", r"\bmove\w*\s+(?:toward|towards|away|across|into|out of|closer|back)\b"),
        (
            "manipulating an object",
            r"\b(?:pick\w*\s+up|put\w*\s+down|set\w*\s+down|hold\w*\s+up|grab\w*|grasp\w*|seiz\w*|snatch\w*"
            r"|pour\w*|drink\w*|sip\w*|eat\w*|typ\w+|writ\w+|draw\w+|adjusts?\b|straighten\w*|remov\w*|unfold\w*)\b",
        ),
        ("manipulating an object", r"\b(?:open\w*|clos\w*|lift\w*|push\w*|pull\w*|turn\w*)\s+(?:the|a|an|her|his|their)\s+\w+"),
        ("manipulating an object", r"\b(?:take\w*|put\w*)\s+(?:off|on)\b"),
    )
)

# --------------------------------------------------------------------------
# Rule 4 — viewer-relative geometry stays viewer-relative.
#
# "her left hand" asserts which way the subject faces. A photograph shows only
# which side of the frame the hand is on, so the subject-relative form is a
# claim the image did not make unless the IR confirms laterality.
# --------------------------------------------------------------------------
SUBJECT_LATERALITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:her|his|their|its|the subject'?s?|the model'?s?|the woman'?s?|the man'?s?|the person'?s?)\s+"
        r"(?:own\s+)?(?:left|right)\b",
        r"\b(?:left|right)\s+(?:hand|arm|leg|foot|knee|shoulder|eye|ear|cheek|side|hip|elbow|wrist|ankle|thigh|palm)\b",
        r"\bstage\s+(?:left|right)\b",
        r"\bto\s+(?:her|his|their)\s+(?:left|right)\b",
    )
)

# The legal form. Offered as the rewrite whenever the subject-relative form is
# rejected, so the fix is in the report rather than left to the reader.
VIEWER_LATERALITY_HINT = "viewer-left / viewer-right (or frame-left / frame-right)"

VIEWER_LATERALITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bviewer[-\s](?:left|right)\b",
        r"\bframe[-\s](?:left|right)\b",
        r"\bcamera[-\s](?:left|right)\b",
        r"\bimage[-\s](?:left|right)\b",
        r"\bscreen[-\s](?:left|right)\b",
        r"\bon the (?:viewer|frame|camera|image|screen)'?s? (?:left|right)\b",
    )
)

# --------------------------------------------------------------------------
# Rules 3 and 7 — specifics that would resolve an uncertain attribute.
#
# The IR's own `candidates` are the first line of defence and cover the readings
# the IR author actually weighed. This table is the backstop for the ones they
# did not write down: if an attribute of this kind is uncertain, no word from
# its row may appear at all. The key is matched as a substring of the slot name,
# so `top_material` and `skirt_material` both pick up `material`.
# --------------------------------------------------------------------------
SLOT_KIND_SPECIFICS: dict[str, tuple[str, ...]] = {
    "material": (
        "satin", "silk", "velvet", "leather", "denim", "chiffon", "linen", "wool", "cotton",
        "lace", "mesh", "latex", "suede", "cashmere", "corduroy", "tweed", "nylon", "polyester",
        "fur", "organza", "taffeta", "jersey", "canvas", "sequin", "sequined", "beaded",
    ),
    "fabric": (
        "satin", "silk", "velvet", "leather", "denim", "chiffon", "linen", "wool", "cotton",
        "lace", "mesh", "latex", "suede", "cashmere", "corduroy", "tweed",
    ),
    "pattern": (
        "striped", "stripes", "plaid", "tartan", "floral", "polka", "houndstooth", "paisley",
        "checkered", "checked", "herringbone", "argyle", "gingham", "leopard", "zebra",
    ),
    "color": (
        "crimson", "scarlet", "burgundy", "maroon", "emerald", "teal", "turquoise", "magenta",
        "mauve", "lavender", "ochre", "mustard", "olive", "navy", "indigo", "cobalt", "cerulean",
        "ivory", "cream", "beige", "taupe", "charcoal", "coral", "salmon", "lilac",
    ),
    "metal": ("gold", "silver", "platinum", "brass", "copper", "bronze", "steel", "titanium"),
    "brand": ("logo", "branded", "monogram", "trademark", "label reading"),
    "text": ("reads", "spelling", "inscribed", "lettering that says"),
    "age": ("teenage", "teenager", "twenties", "thirties", "forties", "fifties", "sixties", "elderly", "middle-aged"),
    "expression": ("smiling", "smirking", "grinning", "frowning", "scowling", "pouting", "laughing", "crying"),
    "emotion": ("happy", "sad", "angry", "afraid", "surprised", "disgusted", "joyful", "melancholy", "anxious"),
    "species": ("labrador", "poodle", "siamese", "tabby", "retriever", "husky", "beagle"),
    "location": ("paris", "tokyo", "london", "manhattan", "kyoto", "venice", "berlin"),
}

# --------------------------------------------------------------------------
# Rule 2 — observed attributes cannot be restated as something else.
#
# Each group holds mutually exclusive poles. When the IR observed one pole and
# the prompt asserts another from the same group, the prompt has changed an
# observed attribute, however fluent the sentence.
# --------------------------------------------------------------------------
EXCLUSIVE_GROUPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "gaze direction",
        (
            ("toward camera", "towards camera", "at the camera", "into the lens", "eye contact", "into the camera", "at the viewer"),
            ("away from camera", "looking away", "looks away", "look away", "averted", "off camera", "off-camera",
             "downcast", "over the shoulder", "into the distance"),
            ("eyes closed", "closed eyes", "eyes shut"),
        ),
    ),
    (
        "posture",
        (
            ("standing", "stands", "upright on her feet", "upright on his feet", "on her feet", "on his feet"),
            ("seated", "sitting", "sits", "perched on"),
            ("lying", "lying down", "reclining", "supine", "prone"),
            ("kneeling", "crouching", "squatting"),
        ),
    ),
    (
        "body orientation",
        (
            ("facing the camera", "facing camera", "front-facing", "frontal"),
            ("in profile", "side profile", "profile view"),
            ("back to the camera", "facing away", "from behind", "rear view"),
        ),
    ),
    (
        "shot size",
        (
            ("close-up", "closeup", "tight on the face", "head shot", "headshot"),
            ("medium shot", "waist up", "half body"),
            ("full body", "full-length", "head to toe"),
            ("wide shot", "long shot", "establishing shot"),
        ),
    ),
    (
        "camera height",
        (
            ("eye level", "eye-level"),
            ("low angle", "from below", "worm's eye"),
            ("high angle", "from above", "overhead", "bird's eye", "top-down"),
        ),
    ),
    (
        "setting",
        (("indoors", "indoor", "interior", "inside a room"), ("outdoors", "outdoor", "exterior", "in the open air")),
    ),
    (
        "time of day",
        (
            ("daylight", "daytime", "midday", "afternoon light", "morning light"),
            ("night", "nighttime", "after dark", "moonlight"),
            ("golden hour", "sunset", "sunrise", "dusk", "twilight"),
        ),
    ),
    (
        "lighting quality",
        (("soft light", "diffused", "overcast", "even light"), ("hard light", "harsh light", "direct sun", "strong shadows")),
    ),
    (
        "hair length",
        (("long hair", "waist-length hair", "shoulder-length hair"), ("short hair", "cropped hair", "buzz cut"), ("bald", "shaved head")),
    ),
    (
        "subject count",
        (("alone", "a single figure", "one person", "solo"), ("two people", "a pair", "a couple"), ("a group", "a crowd", "several people")),
    ),
)

# --------------------------------------------------------------------------
# Paraphrases that say exactly what an observed wording already says.
#
# Without this, tracing is literal: an IR recording `gaze = toward camera`
# would reject "holds eye contact with the camera" as a new fact, because
# "eye" and "contact" are not in the IR's words. That is the wrong verdict —
# the clause restates the observation — and a rule that cries wolf on correct
# prompts gets switched off.
#
# The bar for an entry is equality, not similarity: the paraphrase must be true
# in exactly the cases the canonical wording is true, so admitting it can never
# sharpen, widen or shift the attribute. "Eye contact" qualifies; "gazing
# warmly" does not, because warmth is an attribute the IR never recorded. A
# clause admitted this way still traces to the fact that licensed it, so the
# report names the observation behind every word of it.
# --------------------------------------------------------------------------
EQUIVALENCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("toward camera", ("eye contact", "at the camera", "into the lens", "at the lens", "at the viewer", "toward the lens")),
    ("towards camera", ("eye contact", "at the camera", "into the lens", "at the viewer")),
    ("at the camera", ("eye contact", "toward camera", "into the lens")),
    ("seated", ("sitting", "sits")),
    ("standing", ("stands", "on their feet")),
    ("medium shot", ("waist up", "half body")),
    ("close-up", ("closeup", "tight framing")),
    ("full body", ("full-length", "head to toe")),
    ("eye level", ("level with the eyes",)),
    ("soft light", ("diffused light", "soft lighting")),
    ("hard light", ("harsh light", "hard lighting")),
    ("indoors", ("interior", "inside")),
    ("outdoors", ("exterior", "outside")),
    ("shoulder-length", ("shoulder length",)),
)


# --------------------------------------------------------------------------
# Wording that makes no claim about what the image shows.
#
# Render settings, durations and delivery formats describe the output file, not
# the photograph, so they need no IR trace. Style words are listed apart because
# they do change the picture: they are allowed through, and reported, so the
# author can see exactly which of them a "grounded" prompt is carrying.
# --------------------------------------------------------------------------
TECHNICAL_TERMS: frozenset[str] = frozenset(
    """
    fps frame frames framerate resolution 1080p 1440p 2160p 4k 8k hd uhd aspect ratio
    16 9 4 3 1 seconds second duration loop looping seamless static locked off tripod
    no cut cuts single take continuous shot steady handheld stabilised stabilized
    codec bitrate mp4 webm prores render output upscale upscaled interpolation
    negative prompt weight strength seed steps cfg sampler scheduler denoise
    """.split()
)

STYLE_TERMS: frozenset[str] = frozenset(
    """
    photoreal photorealistic realistic cinematic filmic documentary editorial photographic
    live action animated animation
    grain grainy analog analogue film digital sharp crisp detailed highly quality masterpiece
    bokeh depth field shallow anamorphic 35mm 50mm 85mm lens
    """.split()
)
