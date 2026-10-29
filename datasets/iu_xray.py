"""IU Chest X-Ray / NLM Open-i loader.

Pure Python — no network. Parses the XML reports and gives an index you can
iterate. Image URLs are constructed from the parentImage ids using the
documented Open-i path pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from defusedxml import ElementTree as ET  # safe against XXE / billion-laughs

OPENI_IMG_BASE = "https://openi.nlm.nih.gov/imgs/512"


@dataclass
class IUReport:
    report_id: str
    parent_image_ids: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)  # FINDINGS / IMPRESSION / INDICATION / COMPARISON
    mesh_major: list[str] = field(default_factory=list)
    mesh_automatic: list[str] = field(default_factory=list)


def parse_xml(xml_text: str) -> IUReport:
    """Parse one IU X-Ray report XML string."""
    root = ET.fromstring(xml_text)

    uid_el = root.find(".//uId")
    report_id = uid_el.get("id") if uid_el is not None and uid_el.get("id") else "unknown"

    parent_ids = [el.get("id") for el in root.findall(".//parentImage") if el.get("id")]

    sections: dict[str, str] = {}
    for at in root.findall(".//AbstractText"):
        label = (at.get("Label") or "").upper()
        if label and at.text and at.text.strip():
            sections[label] = at.text.strip()

    mesh_major = [el.text.strip() for el in root.findall(".//MeSH/major")
                  if el.text and el.text.strip()]
    mesh_auto = [el.text.strip() for el in root.findall(".//MeSH/automatic")
                 if el.text and el.text.strip()]

    return IUReport(
        report_id=report_id, parent_image_ids=parent_ids, sections=sections,
        mesh_major=mesh_major, mesh_automatic=mesh_auto,
    )


def load_reports(reports_dir: Path | str) -> list[IUReport]:
    reports_dir = Path(reports_dir)
    out: list[IUReport] = []
    for p in sorted(reports_dir.glob("*.xml")):
        try:
            out.append(parse_xml(p.read_text(encoding="utf-8")))
        except ET.ParseError:
            continue  # skip malformed xmls; never silently claim success elsewhere
    return out


def image_url(parent_image_id: str) -> str:
    """Open-i's documented thumbnail/image URL pattern."""
    # pattern: https://openi.nlm.nih.gov/imgs/512/<CASE>/<PARENT_IMAGE>.png
    # shortcut: assumes the 512 variant; upgrade to a size negotiation if a job needs higher res.
    case = parent_image_id.split("_", 1)[0]  # e.g. "CXR1_IM-0001-1001" -> "CXR1"
    return f"{OPENI_IMG_BASE}/{case}/{parent_image_id}.png"


def report_text(r: IUReport) -> str:
    """Concatenate the common sections into a single auditable report string."""
    parts = []
    for sec in ("INDICATION", "COMPARISON", "FINDINGS", "IMPRESSION"):
        if sec in r.sections:
            parts.append(f"{sec}: {r.sections[sec]}")
    return "\n".join(parts)
