import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    PhaseMixturePolicy,
    PhaseMixtureSession,
)
from braggcalculator.experimental_profile import caglioti_fwhm, render_pseudo_voigt


def _phase_structure(kind):
    if kind == "NaCl":
        return Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
    return Structure(Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _unit_profile(structure, grid, wavelength):
    calculator = BraggCalculator(
        wavelength=wavelength,
        two_theta_range=(float(grid[0]), float(grid[-1])),
        two_theta_step=float(np.median(np.diff(grid))),
        primitive=False,
    ).load(structure)
    centers, areas = calculator.line_components([wavelength], domain="two_theta")[0]
    widths = caglioti_fwhm(np.radians(centers), 0.0025, 1e-6, 0.0064, calculator.backend)
    profile = render_pseudo_voigt(grid, centers, areas, widths, 0.5, calculator.backend)
    return profile / (np.sum(profile) * np.median(np.diff(grid)))


def test_fixed_structure_phase_mixture_recovers_profile_area_fractions_and_report(tmp_path):
    pytest.importorskip("torch")
    wavelength = 1.5406
    grid = np.arange(20.0, 90.0001, 0.06)
    phases = (_phase_structure("NaCl"), _phase_structure("CsCl"))
    unit_profiles = [_unit_profile(phase, grid, wavelength) for phase in phases]
    target = np.array([0.72, 0.28])
    intensity = (
        2500.0 * sum(fraction * profile for fraction, profile in zip(target, unit_profiles)) + 5.0
    )
    dataset = DiffractionDataset(
        coordinate=grid,
        intensity=intensity,
        sigma=np.full(len(grid), 2.0),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=wavelength,
        metadata={"kind": "synthetic two-phase profile-area fraction recovery"},
    )
    policy = PhaseMixturePolicy(
        initial_fractions=(0.5, 0.5),
        refine_profile=False,
        profile_model="legacy",
        background_degree=0,
        diagnostic_points=16,
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 40, 0.03),
            OptimizationStage("phase fractions", ("phase_fractions",), 100, 0.025),
            OptimizationStage("joint", ("scale", "background", "phase_fractions"), 140, 0.008),
        ),
    )
    session = PhaseMixtureSession(dataset, phases, names=("NaCl", "CsCl"))
    result = session.run(policy)

    assert sum(result.phase_fractions.values()) == pytest.approx(1.0, abs=1e-14)
    assert result.phase_fractions["NaCl"] == pytest.approx(target[0], abs=5e-4)
    assert result.phase_fractions["CsCl"] == pytest.approx(target[1], abs=5e-4)
    assert result.r_wp < 2e-3
    assert result.provenance["fraction_definition"] == "integrated profile area over fitted range"
    assert any("not quantitative mass fractions" in warning for warning in result.warnings)
    report = session.write_html(result, tmp_path / "mixture.html")
    assert "Profile-area fraction" in report.read_text(encoding="utf-8")


def test_weak_minor_phase_emits_detectability_warning():
    pytest.importorskip("torch")
    wavelength = 1.5406
    grid = np.arange(20.0, 90.0001, 0.08)
    phases = (_phase_structure("NaCl"), _phase_structure("CsCl"))
    unit_profiles = [_unit_profile(phase, grid, wavelength) for phase in phases]
    fractions = np.array([0.9997, 0.0003])
    components = [
        2500.0 * fraction * profile for fraction, profile in zip(fractions, unit_profiles)
    ]
    intensity = components[0] + components[1] + 5.0
    sigma_level = np.linalg.norm(components[1]) / 2.0
    dataset = DiffractionDataset(
        coordinate=grid,
        intensity=intensity,
        sigma=np.full(len(grid), sigma_level),
        mask=np.ones(len(grid), dtype=bool),
        domain="two_theta",
        wavelength=wavelength,
    )
    policy = PhaseMixturePolicy(
        initial_fractions=tuple(fractions),
        refine_profile=False,
        profile_model="legacy",
        background_degree=0,
        diagnostic_points=0,
        stages=(OptimizationStage("scale/background", ("scale", "background"), 30, 0.02),),
    )
    result = PhaseMixtureSession(dataset, phases, names=("major", "minor")).run(policy)

    assert result.phase_detectability["major"] > 3.0
    assert result.phase_detectability["minor"] < 3.0
    assert any("Phase minor is below" in warning for warning in result.warnings)
