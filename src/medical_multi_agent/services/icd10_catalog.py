"""Offline CMS FY2027 ICD-10-CM membership and header validation.

This catalog distinguishes complete codes from non-reportable headers. It does
not adjudicate whether a code is justified by a narrative or determine coding
sequencing. Runtime access uses packaged resources and never the network.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from functools import lru_cache
from importlib import resources

_PACKAGE = "medical_multi_agent.data"
_MANIFEST = "icd10cm_fy2027.manifest.json"
# FY2027 includes QA... codes; the second character is no longer always numeric.
_CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4}|[A-Z0-9]{0,4})")


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(resources.files(_PACKAGE).joinpath(_MANIFEST).read_text(encoding="utf-8"))


def catalog_info() -> dict:
    """Return source hashes, release and counts without exposing mutable cache state."""
    return copy.deepcopy(_manifest())


@lru_cache(maxsize=1)
def _entries() -> dict[str, tuple[str, bool]]:
    manifest = _manifest()
    compressed = resources.files(_PACKAGE).joinpath(manifest["catalog"]["filename"]).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["catalog"]["sha256"]:
        raise ValueError("Packaged ICD-10-CM catalog checksum mismatch.")
    payload = gzip.decompress(compressed)
    if hashlib.sha256(payload).hexdigest() != manifest["catalog"]["uncompressed_sha256"]:
        raise ValueError("Packaged ICD-10-CM payload checksum mismatch.")
    catalog = json.loads(payload)
    if catalog["schema_version"] != 1 or catalog["release"] != manifest["release"]:
        raise ValueError("Unsupported packaged ICD-10-CM catalog schema or release.")
    entries = {code: (description, is_complete) for code, description, is_complete in catalog["records"]}
    if len(entries) != manifest["counts"]["total_entries"]:
        raise ValueError("Packaged ICD-10-CM catalog entry count mismatch.")
    return entries


def normalize_icd10_code(value: str) -> str | None:
    """Canonicalize valid dotted/undotted syntax; reject malformed input safely."""
    if not isinstance(value, str):
        return None
    code = value.strip()
    if not code.isascii():
        return None
    code = code.upper()
    if not _CODE_PATTERN.fullmatch(code):
        return None
    undotted = code.replace(".", "")
    return undotted[:3] + "." + undotted[3:] if len(undotted) > 3 else undotted


def _record(code: str, description: str, is_complete: bool) -> dict:
    return {
        "code": code,
        "description": description,
        "is_complete_code": is_complete,
        "catalog_release": _manifest()["release"],
    }


def lookup_code(code: str) -> dict | None:
    """Look up either a complete code or an official header, without guessing."""
    canonical = normalize_icd10_code(code)
    if canonical is None:
        return None
    found = _entries().get(canonical)
    return _record(canonical, *found) if found is not None else None


def validate_code(code: str) -> dict:
    """Return explicit malformed/unknown/header/complete membership status."""
    canonical = normalize_icd10_code(code)
    found = lookup_code(canonical) if canonical is not None else None
    complete = bool(found and found["is_complete_code"])
    if canonical is None:
        reason = "malformed_code"
    elif found is None:
        reason = "unknown_code"
    else:
        reason = "complete_code" if complete else "non_reportable_header"
    return {
        "input_code": code,
        "code": canonical,
        "valid": complete,
        "reason": reason,
        "description": found["description"] if found else None,
        "is_complete_code": complete,
        "catalog_release": _manifest()["release"],
    }


def get_complete_codes(prefix: str, limit: int = 30) -> list[dict]:
    """Return bounded complete-code matches under a canonical code prefix."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    canonical = normalize_icd10_code(prefix)
    if canonical is None or limit == 0:
        return []
    matches = []
    for code, (description, complete) in _entries().items():
        if complete and code.startswith(canonical):
            matches.append(_record(code, description, complete))
            if len(matches) == limit:
                break
    return matches
