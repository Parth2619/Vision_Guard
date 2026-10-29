"""Deterministic, rule-based claim extraction.

Implements the ClaimExtractor protocol so an LLM backend can swap in later
without touching the graph or verifier.

shortcut: rule-based only; drop in an LLM extractor by writing a new class with
the same `.extract(text) -> list[Claim]` method and selecting it in the graph.
"""
from __future__ import annotations

import re
from typing import Iterable, Protocol

from .claims import Claim, ClaimType, Polarity

# Small, honest keyword vocabularies. Not clinically exhaustive.
FINDINGS = {
    "pneumonia", "effusion", "pleural effusion", "cardiomegaly", "edema",
    "consolidation", "nodule", "mass", "opacity", "atelectasis",
    "pneumothorax", "fracture", "infiltrate", "emphysema",
}
ANATOMY = {
    "lung", "lungs", "left lung", "right lung", "heart", "mediastinum",
    "pleura", "diaphragm", "ribs", "spine", "bones", "cardiac silhouette",
}
NEGATIONS = ("no ", "no evidence of ", "without ", "negative for ", "ruled out ", "absent ")
UNCERTAIN = ("possible ", "possibly ", "likely ", "suggestive of ", "may represent ",
             "cannot exclude ", "cannot rule out ", "questionable ", "suspicious for ")

SECTION_RE = re.compile(r"^\s*(FINDINGS|IMPRESSION|INDICATION|COMPARISON)\s*:\s*",
                        re.IGNORECASE | re.MULTILINE)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

ATTRIBUTES = ("small", "large", "mild", "moderate", "severe", "massive",
              "minimal", "subtle", "extensive")
SPATIAL = ("upper lobe", "lower lobe", "middle lobe", "apex", "base",
           "left", "right", "bilateral", "central", "peripheral")
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "multiple": -1, "several": -1, "numerous": -1}
COUNT_RE = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                      r"multiple|several|numerous)\b")


class ClaimExtractor(Protocol):
    def extract(self, text: str) -> list[Claim]: ...


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    parts: list[tuple[str | None, str]] = []
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [(None, text)]
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append((name, text[start:end]))
    # Preamble (text before first section) kept if non-empty.
    pre = text[: matches[0].start()].strip()
    if pre:
        parts.insert(0, (None, pre))
    return parts


def _polarity(sent_lower: str) -> tuple[Polarity, str | None]:
    for marker in UNCERTAIN:
        if marker in sent_lower:
            return Polarity.UNCERTAIN, f"hedged by '{marker.strip()}'"
    for marker in NEGATIONS:
        if marker in sent_lower:
            return Polarity.NEGATIVE, f"negated by '{marker.strip()}'"
    return Polarity.POSITIVE, None


def _find_terms(sent_lower: str, vocab: Iterable[str]) -> str | None:
    # Prefer longer matches so "pleural effusion" beats "effusion".
    hits = [t for t in vocab if t in sent_lower]
    if not hits:
        return None
    return max(hits, key=len)


def _classify(lower: str) -> tuple[ClaimType, dict]:
    """Decide claim type and pull out the type-specific fields."""
    extras: dict = {}
    spatial = _find_terms(lower, SPATIAL)
    if spatial:
        extras["spatial_relation"] = spatial
    count_match = COUNT_RE.search(lower)
    if count_match:
        tok = count_match.group(1)
        extras["count"] = int(tok) if tok.isdigit() else NUMBER_WORDS[tok]
        return ClaimType.COUNT, extras
    attr = _find_terms(lower, ATTRIBUTES)
    if attr:
        extras["attribute"] = attr
        return ClaimType.ATTRIBUTE, extras
    if spatial:
        return ClaimType.SPATIAL, extras
    return ClaimType.OBJECT, extras


def extract(text: str) -> list[Claim]:
    """Extract atomic claims from a free-text report. Deterministic."""
    claims: list[Claim] = []
    cursor = 0
    for section, body in _split_sections(text):
        for sentence in SENT_SPLIT.split(body.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            start = text.find(sentence, cursor)
            cursor = start + len(sentence) if start >= 0 else cursor
            lower = sentence.lower()
            finding = _find_terms(lower, FINDINGS)
            if finding is None:
                continue  # no auditable medical finding in this sentence
            polarity, note = _polarity(lower)
            claim_type, extras = _classify(lower)
            claims.append(
                Claim(
                    sentence=sentence,
                    span=(start, start + len(sentence)) if start >= 0 else None,
                    finding=finding,
                    anatomy=_find_terms(lower, ANATOMY),
                    polarity=polarity,
                    claim_type=claim_type,
                    section=section,
                    uncertainty_note=note,
                    **extras,
                )
            )
    return claims


class RuleBasedExtractor:
    """Default extractor; satisfies the ClaimExtractor protocol."""

    def extract(self, text: str) -> list[Claim]:
        return extract(text)
