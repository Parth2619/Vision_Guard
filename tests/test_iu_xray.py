"""IU X-Ray parser tests — offline, uses an inline XML fixture."""
from __future__ import annotations

from datasets.iu_xray import image_url, parse_xml, report_text

_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<eCitation>
  <uId id="42"/>
  <MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText Label="COMPARISON">None.</AbstractText>
        <AbstractText Label="INDICATION">Cough.</AbstractText>
        <AbstractText Label="FINDINGS">There is a small right pleural effusion. No pneumothorax.</AbstractText>
        <AbstractText Label="IMPRESSION">Small right pleural effusion.</AbstractText>
      </Abstract>
    </Article>
  </MedlineCitation>
  <MeSH>
    <major>Pleural Effusion</major>
    <automatic>lung</automatic>
    <automatic>chest</automatic>
  </MeSH>
  <parentImage id="CXR42_IM-0001-1001"/>
  <parentImage id="CXR42_IM-0001-2001"/>
</eCitation>
"""


def test_parse_basic_fields():
    r = parse_xml(_SAMPLE)
    assert r.report_id == "42"
    assert r.parent_image_ids == ["CXR42_IM-0001-1001", "CXR42_IM-0001-2001"]
    assert "FINDINGS" in r.sections and "pleural effusion" in r.sections["FINDINGS"].lower()
    assert "IMPRESSION" in r.sections
    assert r.mesh_major == ["Pleural Effusion"]
    assert r.mesh_automatic == ["lung", "chest"]


def test_report_text_concatenates_in_order():
    text = report_text(parse_xml(_SAMPLE))
    assert text.startswith("INDICATION:")
    assert "FINDINGS:" in text and "IMPRESSION:" in text


def test_image_url_builds_from_parent_id():
    assert image_url("CXR42_IM-0001-1001") == (
        "https://openi.nlm.nih.gov/imgs/512/CXR42/CXR42_IM-0001-1001.png")


def test_parse_handles_missing_sections():
    minimal = ("<?xml version='1.0'?><eCitation><uId id='7'/>"
               "<parentImage id='CXR7_IM-0001-1001'/></eCitation>")
    r = parse_xml(minimal)
    assert r.report_id == "7"
    assert r.parent_image_ids == ["CXR7_IM-0001-1001"]
    assert r.sections == {}
