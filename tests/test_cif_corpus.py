import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from benchmarks.benchmark_cif_corpus import benchmark_case, summarize_results
from benchmarks.cif_corpus import compare_case, load_corpus
from braggcalculator import BraggCalculator


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


def test_corpus_benchmark_times_and_validates_one_structure(tmp_path):
    _, cases = load_corpus(_manifest(tmp_path))
    structure = Structure.from_file(cases[0].path)
    calculator = BraggCalculator(mode="xray")
    oracle = XRDCalculator(wavelength=calculator.wavelength)

    result = benchmark_case(
        cases[0],
        structure,
        "xray",
        calculator=calculator,
        oracle=oracle,
        bragg_first=True,
    )

    assert result["passed"]
    assert result["timing_order"] == ["braggcalculator", "pymatgen"]
    assert result["braggcalculator_seconds"] > 0
    assert result["pymatgen_seconds"] > 0
    assert result["speedup"] == pytest.approx(
        result["pymatgen_seconds"] / result["braggcalculator_seconds"]
    )


def test_corpus_summary_uses_ratio_of_total_runtimes():
    results = [
        {
            "id": "small",
            "braggcalculator_seconds": 1.0,
            "pymatgen_seconds": 4.0,
            "speedup": 4.0,
            "braggcalculator_peaks": 10,
            "passed": True,
        },
        {
            "id": "large",
            "braggcalculator_seconds": 9.0,
            "pymatgen_seconds": 18.0,
            "speedup": 2.0,
            "braggcalculator_peaks": 90,
            "passed": True,
        },
    ]

    summary = summarize_results(results)

    assert summary["total_corpus_speedup"] == pytest.approx(2.2)
    assert summary["median_per_structure_speedup"] == pytest.approx(3.0)
    assert summary["structure_mode_evaluations"] == 2
    assert summary["unique_structures"] == 2


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
