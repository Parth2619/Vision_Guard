import pytest

from datasets.radgenome_schema import EXPECTED_COLUMNS, validate_columns, validate_record


def test_all_expected_present():
    r = {"VolumeName": "v1", "Anatomy": "lung", "Sentence": "no acute findings"}
    chk = validate_record(r)
    assert chk.ok
    assert set(chk.present) == set(EXPECTED_COLUMNS)
    assert chk.missing == ()


def test_case_insensitive_match():
    # HF sometimes serves "Volumename"; our validator must not false-fail on that.
    chk = validate_columns(["Volumename", "anatomy", "sentence"])
    assert chk.ok
    assert chk.missing == ()


def test_missing_column_reported():
    chk = validate_columns(["VolumeName", "Anatomy"])
    assert not chk.ok
    assert "Sentence" in chk.missing


def test_extra_columns_surface_but_do_not_fail():
    chk = validate_columns(["VolumeName", "Anatomy", "Sentence", "ReportID"])
    assert chk.ok
    assert "ReportID" in chk.extra


def test_non_dict_record_raises():
    with pytest.raises(TypeError):
        validate_record(["not", "a", "dict"])  # type: ignore[arg-type]
