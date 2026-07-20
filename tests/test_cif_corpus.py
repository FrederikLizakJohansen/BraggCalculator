import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.cif_corpus import compare_case, load_corpus


ROOT = Path(__file__).resolve().parents[1]


def _manifest(tmp_path, *, digest=None):
    cif = ROOT / "demo" / "NaCl.cif"
    digest = digest or hashlib.sha256(cif.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "name": "test corpus",
        "source": {"name": "local test"},
        "cases": [
            {
                "id": "NaCl",
                "path": str(cif),
                "sha256": digest,
                "source_url": "https://example.invalid/NaCl.cif",
                "crystal_system": "cubic",
                "space_group_number": 225,
                "disordered": False,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_corpus_case_matches_pymatgen(tmp_path):
    _, cases = load_corpus(_manifest(tmp_path))
    result = compare_case(cases[0], "xray", position_atol=1e-10, intensity_atol=1e-9)
    assert result["passed"]
    assert result["braggcalculator_peaks"] == result["pymatgen_peaks"]
    assert result["detected_space_group_number"] == 225
    assert isinstance(result["parser_warnings"], list)


def test_corpus_rejects_changed_cif(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_corpus(_manifest(tmp_path, digest="0" * 64))


def test_frozen_corpus_is_balanced_and_intact():
    manifest, cases = load_corpus(ROOT / "data" / "cif_validation" / "manifest.json")
    assert len(cases) == 70
    assert Counter(case.crystal_system for case in cases) == {
        "triclinic": 10,
        "monoclinic": 10,
        "orthorhombic": 10,
        "tetragonal": 10,
        "trigonal": 10,
        "hexagonal": 10,
        "cubic": 10,
    }
    assert len({case.space_group_number for case in cases}) == 62
    assert sum(case.disordered for case in cases) == 31
    assert manifest["source"]["license"] == "CC0-1.0"
