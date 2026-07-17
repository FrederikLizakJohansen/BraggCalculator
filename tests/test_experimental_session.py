from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import (
    BraggCalculator,
    DiffractionDataset,
    OptimizationStage,
    RefinementPolicy,
    RefinementSession,
    nist_copper_ka_spectrum,
)
from braggcalculator.backends import NumpyBackend
from braggcalculator.experimental_profile import (
    axial_divergence_widths,
    render_pseudo_voigt,
    render_split_pseudo_voigt,
    specimen_displacement_shift,
    thompson_cox_hastings,
)


ROOT = Path(__file__).resolve().parents[1]


def test_xye_ingestion_preserves_uncertainty_mask_and_provenance(tmp_path):
    path = tmp_path / "pattern.xye"
    path.write_text("# x y w\n10,100,0.25\n11,121,0.20\n12,144,0.16\n")
    dataset = DiffractionDataset.from_xye(
        path, wavelength=1.54, third_column="weight"
    )
    np.testing.assert_allclose(dataset.sigma, [2.0, np.sqrt(5.0), 2.5])
    assert dataset.source_sha256 is not None
    assert dataset.step == pytest.approx(1.0)
    excluded = dataset.exclude([(10.5, 11.5)])
    np.testing.assert_array_equal(excluded.mask, [True, False, True])


def test_pseudo_voigt_is_area_normalized():
    backend = NumpyBackend()
    grid = np.linspace(-5, 5, 20001)
    profile = render_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.2]),
        0.35,
        backend,
    )
    assert np.trapezoid(profile, grid) == pytest.approx(3.0, rel=0.01)


def test_split_pseudo_voigt_is_area_normalized_and_has_symmetric_limit():
    backend = NumpyBackend()
    grid = np.linspace(-10, 10, 40001)
    split = render_split_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.3]),
        np.array([0.1]),
        np.array([0.35]),
        backend,
    )
    assert np.trapezoid(split, grid) == pytest.approx(3.0, rel=0.01)
    symmetric = render_split_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.2]),
        np.array([0.2]),
        np.array([0.35]),
        backend,
    )
    reference = render_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.2]),
        np.array([0.35]),
        backend,
    )
    np.testing.assert_allclose(symmetric, reference, rtol=1e-13, atol=1e-13)


def test_instrument_profile_limits_and_specimen_shift():
    backend = NumpyBackend()
    angle = np.radians(np.array([20.0, 80.0]))
    fwhm, eta = thompson_cox_hastings(
        angle, 0.002, -0.0001, 0.004, 0.01, 0.005, backend
    )
    assert np.all(fwhm > 0)
    assert np.all((eta >= 0) & (eta <= 1))
    low, high = axial_divergence_widths(fwhm, angle, 0.0, backend)
    np.testing.assert_allclose(low, high)
    shift = specimen_displacement_shift(angle, -0.08, 200.0, backend)
    expected = np.degrees(0.0008 * np.cos(angle / 2.0))
    np.testing.assert_allclose(shift, expected)


def test_instrument_profile_recovers_synthetic_parameters_and_gradient():
    torch = pytest.importorskip("torch")
    from braggcalculator.backends import TorchBackend

    backend = TorchBackend()
    grid = torch.linspace(15.0, 80.0, 3251, dtype=torch.float64)
    centers = torch.tensor([25.0, 45.0, 70.0], dtype=torch.float64)
    amplitudes = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64)
    radians = torch.deg2rad(centers)

    def calculate(raw):
        fwhm, eta = thompson_cox_hastings(
            radians,
            0.0025 * torch.exp(raw[0]),
            0.0,
            0.0036 * torch.exp(raw[1]),
            0.01 * torch.exp(raw[2]),
            0.006,
            backend,
        )
        low, high = axial_divergence_widths(
            fwhm, radians, 0.05 * torch.exp(raw[3]), backend
        )
        return render_split_pseudo_voigt(
            grid, centers, amplitudes, low, high, eta, backend
        )

    target_raw = torch.tensor([0.3, -0.2, 0.4, 0.25], dtype=torch.float64)
    target = calculate(target_raw).detach()
    raw = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=0.04)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.mean((calculate(raw) - target) ** 2)
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(raw, target_raw, atol=2e-4, rtol=0)

    probe = target_raw.clone().requires_grad_(True)
    objective = torch.sum(calculate(probe) * grid)
    gradient = torch.autograd.grad(objective, probe)[0][3].item()
    step = 1e-5
    upper = target_raw.clone()
    lower = target_raw.clone()
    upper[3] += step
    lower[3] -= step
    finite_difference = (
        torch.sum(calculate(upper) * grid) - torch.sum(calculate(lower) * grid)
    ).item() / (2 * step)
    assert gradient == pytest.approx(finite_difference, rel=2e-5, abs=2e-5)


def test_multiline_components_share_exact_structure_factor(nacl):
    wavelengths = (1.5405925, 1.5443873)
    calculator = BraggCalculator(
        wavelength=wavelengths[0], two_theta_range=(20.0, 70.0), primitive=False
    ).load(nacl)
    components = calculator.iq_components(wavelengths)
    for wavelength, (positions, intensities) in zip(wavelengths, components):
        reference = BraggCalculator(
            wavelength=wavelength, two_theta_range=(20.0, 70.0), primitive=False
        ).load(nacl)
        expected_positions, expected_intensities = reference.iq()
        np.testing.assert_allclose(positions, expected_positions, atol=1e-12)
        np.testing.assert_allclose(intensities, expected_intensities, rtol=1e-12)
    lines = calculator.line_components(wavelengths)
    for (_, point_intensity), (_, line_intensity) in zip(components, lines):
        assert np.sum(line_intensity) == pytest.approx(np.sum(point_intensity))


def test_candidate_session_ranks_generating_structure_and_writes_report(tmp_path, nacl):
    generator = BraggCalculator(
        two_theta_range=(20.0, 60.0), two_theta_step=0.05
    ).load(nacl)
    coordinate, profile = generator.pattern()
    observed = 0.002 * profile + 10.0
    sigma = np.sqrt(np.maximum(observed, 1.0))
    dataset = DiffractionDataset(
        coordinate=coordinate,
        intensity=observed,
        sigma=sigma,
        mask=np.ones(len(coordinate), dtype=bool),
        domain="two_theta",
        wavelength=generator.wavelength,
        metadata={"kind": "synthetic candidate regression"},
    )
    competing = Structure(
        Lattice.cubic(4.2), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    session = RefinementSession(dataset, [nacl, competing], names=["NaCl", "CsCl-like"])
    result = session.run(RefinementPolicy.quick())
    assert result.ranking[0] == "NaCl"
    assert result.pairwise_discrimination
    report = session.write_html(result, tmp_path / "report.html")
    text = report.read_text()
    assert "Diffraction diagnostic report" in text
    assert "Largest unexplained regions" in text
    assert "SHA-256" in text


def test_nist_srm660c_real_data_regression():
    metadata = {
        "wavelength_components": nist_copper_ka_spectrum(),
        "reference_lattice_angstrom": 4.156826,
        "instrument": {"geometry": "Bragg-Brentano", "goniometer_radius_mm": 217.5},
        "model_limitations": ["Compact profile is not the full NIST FPA model."],
    }
    dataset = DiffractionDataset.from_xye(
        ROOT / "data/nist_srm660c_100a_full.xye",
        wavelength=1.5405925,
        metadata=metadata,
    )
    policy = RefinementPolicy(
        background_degree=6,
        specimen_displacement_mm=-0.07877,
        goniometer_radius_mm=217.5,
        diagnostic_points=0,
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 50, 0.03),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 100, 0.01),
            OptimizationStage(
                "joint", ("scale", "background", "profile", "lattice"), 150, 0.003
            ),
        )
    )
    result = RefinementSession(
        dataset, [ROOT / "data/LaB6_srm660c.cif"], names=["LaB6"]
    ).run(policy)
    candidate = result.candidates[0]
    refined_a = candidate.physical_parameters["lattice"][0][0]
    assert candidate.r_wp < 0.16
    assert refined_a == pytest.approx(metadata["reference_lattice_angstrom"], abs=0.001)
    assert candidate.physical_parameters["cell_parameters"]["b"] == pytest.approx(refined_a)
    assert len(candidate.provenance["wavelength_components"]) == 6
    assert candidate.provenance["dataset_sha256"] == dataset.source_sha256
    assert candidate.warnings  # The simplified profile must not claim certification-grade agreement.
