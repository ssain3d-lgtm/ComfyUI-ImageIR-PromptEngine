"""Typed reference roles, with no media loading or second backend stack."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import ImageIR, parse_ir

ROLES = ("identity", "wardrobe", "clothing", "body", "environment", "composition", "motion", "pose", "style", "sound")
MEDIA_ROLES = {"image": set(ROLES) - {"sound"}, "video": {"motion", "composition", "pose", "style"}, "audio": {"sound"}}


@dataclass(frozen=True)
class Reference:
    label: str
    type: str
    role: str
    subject: str = "1"
    image_ir: ImageIR | None = None

    def __post_init__(self):
        prefix = {"image": "Picture", "video": "Video", "audio": "Audio"}.get(self.type)
        if not prefix or not isinstance(self.label, str) or not re.fullmatch(rf"{prefix} [1-9]\d*", self.label):
            raise ValueError("reference label must match its media type, e.g. Picture 1")
        if self.role not in MEDIA_ROLES[self.type]:
            raise ValueError("invalid reference role for media type")
        if not isinstance(self.subject, str) or not re.fullmatch(r"[1-9]\d*", self.subject):
            raise ValueError("reference subject must be a positive numeric string")
        if self.image_ir is not None and (self.type != "image" or not isinstance(self.image_ir, ImageIR)):
            raise ValueError("only image references carry IMAGE_IR")

    def scoped_ir(self) -> ImageIR:
        ir = self.image_ir or ImageIR()
        # Never let an asset's unrelated clothing, room or person enter another role.
        def keep(fact):
            if self.role in ("wardrobe", "clothing"):
                return fact.section == "wardrobe"
            if self.role == "environment":
                return fact.section in ("scene", "lighting")
            if self.role == "composition":
                return fact.section == "camera"
            if self.role in ("pose", "motion"):
                return fact.section == "subject" and fact.slot in ("pose", "hands", "gaze")
            if self.role == "body":
                return fact.section == "subject" and fact.slot in ("body", "build", "height")
            if self.role == "identity":
                return fact.section == "subject" and fact.slot not in ("pose", "hands", "gaze", "body", "build", "height")
            return False  # style/sound are role directives, not a bag of visual facts
        return ImageIR(tuple(f for f in ir.facts if keep(f)), ir.frame, ir.subject_laterality_confirmed)


@dataclass(frozen=True)
class ReferencePack:
    references: tuple[Reference, ...] = ()
    schema: str = "REFERENCE_PACK_v1"

    def __post_init__(self):
        if self.schema != "REFERENCE_PACK_v1" or not isinstance(self.references, tuple):
            raise ValueError("invalid REFERENCE_PACK_v1")
        if any(not isinstance(r, Reference) for r in self.references):
            raise ValueError("references must contain Reference objects")
        labels = [r.label for r in self.references]
        if len(labels) != len(set(labels)):
            raise ValueError("duplicate reference label; combine roles with distinct assets")
        assignments = [(r.subject, "wardrobe" if r.role == "clothing" else r.role) for r in self.references]
        if len(assignments) != len(set(assignments)):
            raise ValueError("ambiguous competing references for the same subject role")

    def to_json(self):
        return json.dumps({"schema": self.schema, "references": [
            {"label": r.label, "type": r.type, "role": r.role, "subject": r.subject,
             "image_ir": r.image_ir.to_dict() if r.image_ir else None} for r in self.references
        ]}, ensure_ascii=False, indent=2)


def parse_reference_pack(value) -> ReferencePack:
    if isinstance(value, ReferencePack):
        return value
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict) or set(data) != {"schema", "references"} or not isinstance(data["references"], list):
        raise ValueError("REFERENCE_PACK requires schema and references array")
    refs = []
    try:
        for entry in data["references"]:
            item = dict(entry)
            if item.get("image_ir") is not None:
                item["image_ir"] = parse_ir(item["image_ir"])
            refs.append(Reference(**item))
        return ReferencePack(tuple(refs), data["schema"])
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid reference entry") from exc
