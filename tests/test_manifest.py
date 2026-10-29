import json

import pytest

from datasets.manifest import (
    DEFAULT_SAMPLE_LIMIT,
    MAX_SAMPLE_LIMIT,
    aggregate_records,
    build_manifest,
    clamp_limit,
    write_manifest,
)


def _rec(vol, anat, sent):
    return {"Volumename": vol, "Anatomy": anat, "Sentence": sent}


def test_aggregate_basic_counts():
    records = [
        _rec("v1", "lung", "clear"),
        _rec("v1", "heart", "normal"),
        _rec("v2", "lung", "nodule 5mm"),
    ]
    agg = aggregate_records(records)
    assert agg.sampled_rows == 3
    assert agg.unique_volumes == 2
    assert agg.unique_anatomies == 2
    assert agg.non_empty_sentences == 3
    assert agg.missing_by_column == {"Volumename": 0, "Anatomy": 0, "Sentence": 0}


def test_aggregate_duplicate_volumes_grouped():
    records = [_rec("v1", "lung", "a"), _rec("v1", "lung", "b"), _rec("v1", "heart", "c")]
    agg = aggregate_records(records)
    assert agg.unique_volumes == 1
    assert agg.per_volume["v1"]["rows"] == 3
    assert set(agg.per_volume["v1"]["unique_anatomies"]) == {"lung", "heart"}


def test_aggregate_missing_values_counted():
    records = [
        _rec(None, "lung", "x"),
        _rec("v2", None, ""),
        _rec("v3", "liver", "   "),
    ]
    agg = aggregate_records(records)
    assert agg.missing_by_column == {"Volumename": 1, "Anatomy": 1, "Sentence": 2}
    assert agg.non_empty_sentences == 1
    # Missing-volume rows don't inflate unique_volumes.
    assert agg.unique_volumes == 2


def test_case_insensitive_keys():
    agg = aggregate_records([{"volumename": "v1", "anatomy": "lung", "sentence": "ok"}])
    assert agg.sampled_rows == 1
    assert agg.unique_volumes == 1


def test_clamp_limit_bounds():
    assert clamp_limit(50) == 50
    with pytest.raises(ValueError):
        clamp_limit(0)
    with pytest.raises(ValueError):
        clamp_limit(MAX_SAMPLE_LIMIT + 1)
    # Explicit override is allowed.
    assert clamp_limit(MAX_SAMPLE_LIMIT + 1, allow_override=True) == MAX_SAMPLE_LIMIT + 1


def test_manifest_excludes_sentences_by_default():
    agg = aggregate_records([_rec("v1", "lung", "patient-identifying-text")])
    m = build_manifest(agg, dataset_id="x", config="grounded reports",
                       sample_limit=DEFAULT_SAMPLE_LIMIT)
    blob = json.dumps(m)
    assert "patient-identifying-text" not in blob
    assert "sentence_preview" not in m


def test_manifest_preview_opt_in_and_truncated():
    agg = aggregate_records([_rec("v1", "lung", "s")])
    long_text = "x" * 200
    m = build_manifest(agg, dataset_id="x", config="grounded reports", sample_limit=10,
                       include_sentence_preview=True, preview_examples=[long_text])
    assert m["sentence_preview"][0].endswith("…")
    assert len(m["sentence_preview"][0]) <= 81


def test_write_manifest_overwrite_protection(tmp_path):
    out = tmp_path / "m.json"
    agg = aggregate_records([_rec("v1", "lung", "s")])
    m = build_manifest(agg, dataset_id="x", config="c", sample_limit=1)

    write_manifest(m, out)
    assert out.exists()

    with pytest.raises(FileExistsError):
        write_manifest(m, out)  # second write must refuse

    # --force path works.
    write_manifest(m, out, force=True)
    loaded = json.loads(out.read_text())
    assert loaded["aggregate"]["sampled_rows"] == 1
    assert "disclaimer" in loaded["meta"]
