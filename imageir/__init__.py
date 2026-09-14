"""The IMAGE_IR grounding engine, importable without ComfyUI.

Kept free of ComfyUI imports on purpose: the rules are the valuable part, and
they are easier to trust when they can be run, tested and argued with from a
plain Python prompt.
"""

from .compose import Clause, Composition, compose
from .dsl import parse_dsl
from .guard import AuditResult, Finding, RULE_TEXT, audit
from .motion import MotionPlan, MotionRejection, available_motions, plan_motion
from .schema import Fact, IRError, ImageIR, MergeConflict, merge_ir, parse_ir

__all__ = [
    "AuditResult",
    "Clause",
    "Composition",
    "Fact",
    "Finding",
    "IRError",
    "ImageIR",
    "MergeConflict",
    "MotionPlan",
    "MotionRejection",
    "RULE_TEXT",
    "audit",
    "available_motions",
    "compose",
    "merge_ir",
    "parse_dsl",
    "parse_ir",
    "plan_motion",
]
