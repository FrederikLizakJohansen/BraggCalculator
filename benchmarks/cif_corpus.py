"""Reproducible CIF-corpus validation against pymatgen diffraction."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from braggcalculator import BraggCalculator


@dataclass(frozen=True)
class CorpusCase:
    """One immutable structure record from a corpus manifest."""

    identifier: str
    path: Path
    sha256: str
    source_url: str
    crystal_system: str
    space_group_number: int
    disordered: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(manifest_path: Path) -> tuple[dict, list[CorpusCase]]:
    """Read and validate a versioned corpus manifest."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("CIF corpus manifest must have schema_version 1")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("CIF corpus manifest must contain at least one case")

    cases = []
    identifiers = set()
    for raw in raw_cases:
        identifier = str(raw["id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate CIF corpus id: {identifier}")
        identifiers.add(identifier)
        path = (manifest_path.parent / raw["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing CIF corpus file: {path}")
        expected_hash = str(raw["sha256"]).lower()
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {identifier}: expected {expected_hash}, got {actual_hash}"
            )
        cases.append(
            CorpusCase(
                identifier=identifier,
                path=path,
                sha256=expected_hash,
                source_url=str(raw["source_url"]),
                crystal_system=str(raw["crystal_system"]),
                space_group_number=int(raw["space_group_number"]),
                disordered=bool(raw["disordered"]),
            )
        )
    return manifest, cases


def compare_case(
    case: CorpusCase,
    mode: str,
    *,
    position_atol: float,
    intensity_atol: float,
) -> dict:
    """Compare one CIF in one diffraction mode and return machine-readable metrics."""
    if mode not in {"xray", "neutron"}:
        raise ValueError("mode must be 'xray' or 'neutron'")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        structure = Structure.from_file(case.path)
    parser_warnings = sorted({str(item.message) for item in caught_warnings})
    calculator = BraggCalculator(mode=mode).load(structure)
    actual_position, actual_intensity = calculator.line_pattern(scaled=True)
    oracle_type = XRDCalculator if mode == "xray" else NDCalculator
    oracle = oracle_type(wavelength=calculator.wavelength).get_pattern(
        structure,
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )
    actual_position = np.asarray(actual_position, dtype=float)
    actual_intensity = np.asarray(actual_intensity, dtype=float)
    expected_position = np.asarray(oracle.x, dtype=float)
    expected_intensity = np.asarray(oracle.y, dtype=float)
    peak_count_matches = actual_position.shape == expected_position.shape
    if peak_count_matches:
        max_position_error = float(
            np.max(np.abs(actual_position - expected_position), initial=0.0)
        )
        max_intensity_error = float(
            np.max(np.abs(actual_intensity - expected_intensity), initial=0.0)
        )
    else:
        max_position_error = None
        max_intensity_error = None
    passed = bool(
        peak_count_matches
        and max_position_error <= position_atol
        and max_intensity_error <= intensity_atol
    )
    return {
        "id": case.identifier,
        "mode": mode,
        "source_url": case.source_url,
        "sha256": case.sha256,
        "crystal_system": case.crystal_system,
        "manifest_space_group_number": case.space_group_number,
        "detected_space_group_number": calculator._symm["spacegroup_number"],
        "disordered": case.disordered,
        "parser_warnings": parser_warnings,
        "formula": structure.composition.formula,
        "input_sites": len(structure),
        "reduced_sites": len(calculator._symm["structure"]),
        "braggcalculator_peaks": int(actual_position.size),
        "pymatgen_peaks": int(expected_position.size),
        "max_position_error_deg": max_position_error,
        "max_intensity_error_percent": max_intensity_error,
        "position_atol_deg": position_atol,
        "intensity_atol_percent": intensity_atol,
        "passed": passed,
    }
