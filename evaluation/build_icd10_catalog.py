"""Rebuild the offline FY2027 ICD-10-CM catalog from the pinned CMS archive.

Download the source archive separately, then run from the repository root:
    python -m evaluation.build_icd10_catalog --archive /path/to/source.zip

This importer uses only the standard library and makes no network requests.
It verifies the archive before parsing or writing any output. Gzip timestamps,
JSON formatting and source ordering are fixed for reproducible output bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

SOURCE_URL = "https://www.cms.gov/files/zip/2027-code-descriptions-tabular-order.zip"
ARCHIVE_SHA256 = "91c6c9d1117764ce72375a0f3a5493b1725dafbc1ab283b55799076c9e194965"
CODES_MEMBER = "Code Descriptions/icd10cm_codes_2027.txt"
ORDER_MEMBER = "Code Descriptions/icd10cm_order_2027.txt"
CATALOG_FILENAME = "icd10cm_fy2027.json.gz"
MANIFEST_FILENAME = "icd10cm_fy2027.manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "src" / "medical_multi_agent" / "data"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_source_code(code: str) -> str:
    if not re.fullmatch(r"[A-Z][A-Z0-9]{2,6}", code):
        raise ValueError("Unexpected code syntax in the authoritative source.")
    return code[:3] + "." + code[3:] if len(code) > 3 else code


def build_catalog(archive: Path, output_dir: Path) -> dict:
    """Verify and import official complete codes plus non-reportable headers."""
    archive_bytes = Path(archive).read_bytes()
    if _sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("Source archive SHA-256 does not match the pinned CMS release.")
    with ZipFile(io.BytesIO(archive_bytes)) as source:
        raw_codes = source.read(CODES_MEMBER)
        raw_order = source.read(ORDER_MEMBER)
    code_lines = raw_codes.decode("utf-8-sig").splitlines()
    order_lines = raw_order.decode("utf-8-sig").splitlines()
    complete = {}
    for line in code_lines:
        code, description = line.split(maxsplit=1)
        canonical = _canonical_source_code(code)
        if canonical in complete or not description:
            raise ValueError("Duplicate or empty complete-code source entry.")
        complete[canonical] = description

    records = []
    seen = set()
    complete_in_order = set()
    for expected_order, line in enumerate(order_lines, 1):
        # CMS order format: sequence(5), code(7), complete-code flag(1),
        # short description(60), long description; fields separated by spaces.
        if len(line) < 78 or int(line[:5]) != expected_order or line[14] not in "01":
            raise ValueError("Unexpected record format or order in the CMS order file.")
        code = _canonical_source_code(line[6:13].strip())
        is_complete = line[14] == "1"
        description = line[77:].rstrip()
        if code in seen or not description:
            raise ValueError("Duplicate or empty order-file entry.")
        seen.add(code)
        if is_complete:
            if complete.get(code) != description:
                raise ValueError("Complete-code membership/description differs between CMS source files.")
            complete_in_order.add(code)
        elif code in complete:
            raise ValueError("A CMS header was also listed as a complete code.")
        records.append([code, description, is_complete])
    if complete_in_order != set(complete):
        raise ValueError("CMS order file does not cover every complete code.")

    payload = (json.dumps({
        "schema_version": 1, "release": "FY2027", "records": records,
    }, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0, compresslevel=9) as stream:
        stream.write(payload)
    catalog_bytes = compressed.getvalue()
    manifest = {
        "schema_version": 1,
        "code_set": "ICD-10-CM",
        "release": "FY2027",
        "source": {
            "url": SOURCE_URL,
            "sha256": ARCHIVE_SHA256,
            "members": {
                CODES_MEMBER: {"sha256": _sha256(raw_codes), "lines": len(code_lines)},
                ORDER_MEMBER: {"sha256": _sha256(raw_order), "lines": len(order_lines)},
            },
        },
        "counts": {
            "complete_codes": len(complete),
            "non_reportable_headers": len(records) - len(complete),
            "total_entries": len(records),
        },
        "catalog": {
            "filename": CATALOG_FILENAME,
            "sha256": _sha256(catalog_bytes),
            "uncompressed_sha256": _sha256(payload),
        },
        "scope": (
            "Official release membership and long descriptions only. Complete-code status "
            "does not establish clinical appropriateness, sequencing or encounter-specific coding."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CATALOG_FILENAME).write_bytes(catalog_bytes)
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="downloaded pinned CMS ZIP archive")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        info = build_catalog(args.archive, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"release": info["release"], **info["counts"], "catalog_sha256": info["catalog"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
