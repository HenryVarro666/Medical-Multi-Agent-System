"""Offline checks of authoritative membership, normalization and reproducible import."""

import hashlib
import json
from importlib import resources
from zipfile import ZipFile

import pytest


@pytest.mark.parametrize(("value", "expected"), [
    (" j18.9 ", "J18.9"), ("i2109", "I21.09"), ("A00", "A00"),
    ("s52501a", "S52.501A"), ("m1a9xx0", "M1A.9XX0"), ("t46.1x5a", "T46.1X5A"),
    ("qa00101", "QA0.0101"),
])
def test_normalizes_official_dotted_and_undotted_forms(value, expected):
    from medical_multi_agent.services.icd10_catalog import normalize_icd10_code

    assert normalize_icd10_code(value) == expected


@pytest.mark.parametrize("value", [
    "", "I21.", "I2.109", "I21..09", "I 21.09", "I21.0 9", "I21-09", "I21.09999",
    "123.45", "ſ90.1", None, {}, 2109,
])
def test_rejects_malformed_codes_without_guessing(value):
    from medical_multi_agent.services.icd10_catalog import normalize_icd10_code, validate_code

    assert normalize_icd10_code(value) is None
    result = validate_code(value)
    assert result["code"] is None
    assert result["input_code"] == value
    assert result["valid"] is False
    assert result["reason"] == "malformed_code"


@pytest.mark.parametrize("code", ["I21.09", "J18.1", "J18.9", "S52.501A", "M1A.9XX0", "T46.1X5A", "QA0.0101"])
def test_recognizes_complete_codes_including_seventh_characters_and_placeholders(code):
    from medical_multi_agent.services.icd10_catalog import lookup_code, validate_code

    lookup = lookup_code(code)
    assert lookup["is_complete_code"] is True
    assert lookup["description"]
    result = validate_code(code)
    assert result["valid"] is True
    assert result["reason"] == "complete_code"
    assert result["catalog_release"] == "FY2027"


def test_preserves_non_reportable_header_distinction():
    from medical_multi_agent.services.icd10_catalog import lookup_code, validate_code

    header = lookup_code("i210")
    assert header["code"] == "I21.0"
    assert header["is_complete_code"] is False
    result = validate_code("I21.0")
    assert result["valid"] is False
    assert result["reason"] == "non_reportable_header"
    assert result["description"] == header["description"]


def test_rejects_unknown_suffix_without_remapping_to_a_known_code():
    from medical_multi_agent.services.icd10_catalog import lookup_code, validate_code

    assert lookup_code("I21.09ZZ") is None
    result = validate_code("I21.09ZZ")
    assert result["code"] == "I21.09ZZ"
    assert result["valid"] is False
    assert result["reason"] == "unknown_code"


def test_complete_code_candidates_are_bounded_and_never_return_headers():
    from medical_multi_agent.services.icd10_catalog import get_complete_codes

    family = get_complete_codes("I21.0")
    assert {entry["code"] for entry in family} == {"I21.01", "I21.02", "I21.09"}
    assert all(entry["is_complete_code"] for entry in family)
    assert len(get_complete_codes("I21", limit=2)) == 2
    assert get_complete_codes("J18.9")[0]["code"] == "J18.9"
    assert get_complete_codes("I21", limit=0) == []
    assert get_complete_codes("not a code") == []
    with pytest.raises(ValueError):
        get_complete_codes("I21", limit=-1)


def test_packaged_catalog_has_full_source_counts_and_verified_provenance():
    from medical_multi_agent.services.icd10_catalog import catalog_info

    info = catalog_info()
    assert info["code_set"] == "ICD-10-CM"
    assert info["release"] == "FY2027"
    assert info["counts"] == {"complete_codes": 74879, "non_reportable_headers": 23524, "total_entries": 98403}
    assert info["source"]["sha256"] == "91c6c9d1117764ce72375a0f3a5493b1725dafbc1ab283b55799076c9e194965"
    blob = resources.files("medical_multi_agent.data").joinpath(info["catalog"]["filename"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == info["catalog"]["sha256"]
    info["counts"]["complete_codes"] = 0
    assert catalog_info()["counts"]["complete_codes"] == 74879


def test_importer_is_reproducible_and_checks_source_membership(tmp_path, monkeypatch):
    from evaluation import build_icd10_catalog as builder

    def order_line(number, code, flag, description):
        return f"{number:05d} {code:<7} {flag} {description:<60} {description}\r\n"

    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as source:
        source.writestr(builder.CODES_MEMBER, "A000    Cholera\r\n")
        source.writestr(builder.ORDER_MEMBER, order_line(1, "A00", 0, "Cholera category") + order_line(2, "A000", 1, "Cholera"))
    monkeypatch.setattr(builder, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    first, second = tmp_path / "first", tmp_path / "second"
    info = builder.build_catalog(archive, first)
    builder.build_catalog(archive, second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert info["counts"] == {"complete_codes": 1, "non_reportable_headers": 1, "total_entries": 2}
    assert (first / builder.CATALOG_FILENAME).read_bytes()[4:8] == b"\0\0\0\0"
    assert json.loads((first / builder.MANIFEST_FILENAME).read_text())["source"]["sha256"] == builder.ARCHIVE_SHA256


def test_importer_rejects_unverified_archive_before_writing(tmp_path):
    from evaluation.build_icd10_catalog import build_catalog

    archive = tmp_path / "untrusted.zip"
    archive.write_bytes(b"not the authoritative archive")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        build_catalog(archive, output)
    assert not output.exists()
