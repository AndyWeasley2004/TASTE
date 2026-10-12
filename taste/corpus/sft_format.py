"""Section-label canonicalization and parsing of the high-quality tier's prose generation prompts."""
from __future__ import annotations

import re
from typing import Any

from taste.representation.tokenizer import SECTION_TOKENS, get_tokenizer


def _single_line(value: Any, name: str) -> str:
    text = " ".join(str(value or "").split())
    if "\n" in text or "\r" in text:
        raise ValueError(f"{name} must be one line")
    return text


def parse_legacy_prompt(
    prompt: str,
    *,
    require_all: bool = True,
) -> dict[str, str]:
    """Extract the five controls from a prose generation prompt."""
    patterns = {
        "emotion": r"^Emotion is (.+?)\.$",
        "genre": r"^Genre is (.+?)\.$",
        "accompaniment": r"^Accompaniment is (.+?)\.$",
        "performance": r"^Performance is (.+?)\.$",
        "imagery": r"^Mood imagery:\s*(.+)$",
    }
    fields: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, prompt, flags=re.MULTILINE)
        if match:
            fields[name] = _single_line(match.group(1), name)
    if require_all and len(fields) != len(patterns):
        raise ValueError(f"could not parse prompt fields: {sorted(fields)}")
    return fields


def canonical_section_ids(sections: list[dict[str, Any]]) -> list[int]:
    """Map source labels to relative ``<SEC_A>`` through ``<SEC_H>`` IDs in order of first appearance."""
    tokenizer = get_tokenizer()
    mapping: dict[str, int] = {}
    output: list[int] = []
    for section in sections:
        label = section.get("letter") or section.get("label") or section.get("name")
        match = re.match(r"^([A-Za-z])", str(label or ""))
        if not match:
            raise ValueError(f"invalid section label: {section}")
        label = match.group(1).upper()
        if label not in mapping:
            if len(mapping) >= len(SECTION_TOKENS):
                raise ValueError("more than 8 distinct section labels")
            mapping[label] = len(mapping)
        output.append(tokenizer.section_ids[mapping[label]])
    return output
