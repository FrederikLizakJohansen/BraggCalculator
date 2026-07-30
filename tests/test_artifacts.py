from hashlib import sha256
import json

import numpy as np
import pytest

from braggcalculator import (
    AmorphousHump,
    BackgroundArtifacts,
    BackgroundLibrary,
    BackgroundPattern,
    BraggCalculator,
    CalibrationArtifacts,
    DetectorArtifacts,
    IntensityArtifacts,
    NoiseArtifacts,
    PeakProfileArtifacts,
    PreferredOrientation,
    SimulationArtifacts,
    SpuriousPeakArtifacts,
)
from braggcalculator.backends import NumpyBackend, TorchBackend
from braggcalculator.artifacts import _render_split_pseudo_voigt


def test_empty_artifact_configuration_preserves_default_pattern(nacl):
    calculator = BraggCalculator(q_step=0.02).load(nacl)
    grid, expected = calculator.pattern(domain="q")
    artifact_grid, actual = calculator.pattern(
        domain="q", artifacts=SimulationArtifacts(clip_nonnegative=False)
    )
    np.testing.assert_array_equal(artifact_grid, grid)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_every_artifact_component_is_independently_configurable():
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(zero_shift=0.01),
        profile=PeakProfileArtifacts(model="pseudo_voigt", fwhm=0.05, eta=0.2),
        intensity=IntensityArtifacts(scale=2.0),
        background=BackgroundArtifacts(constant=0.1),
        noise=NoiseArtifacts(gaussian_std=0.01),
        detector=DetectorArtifacts(saturation_level=1.0),
        spurious_peaks=SpuriousPeakArtifacts(count=1),
        normalize_signal=True,
        final_normalize=True,
        domain="q",
        seed=9,
    )
    assert artifacts.profile.fwhm == 0.05
    assert artifacts.background.constant == 0.1
    assert artifacts.noise.gaussian_std == 0.01


def test_fixed_linear_background_is_added_after_profile_rendering(nacl):
    calculator = BraggCalculator(q_range=(0.0, 1.0), q_step=0.25).load(nacl)
    grid, expected = calculator.pattern(domain="q")
    artifacts = SimulationArtifacts(
        background=BackgroundArtifacts(constant=2.0, linear_slope=0.5),
        clip_nonnegative=False,
    )
    _, actual = calculator.pattern(domain="q", artifacts=artifacts)
    expected_background = 2.0 + 0.5 * (grid - 0.5)
    np.testing.assert_allclose(actual, expected + expected_background, atol=1e-14)


def test_seeded_artifacts_are_reproducible_and_nonnegative(nacl):
    calculator = BraggCalculator(q_step=0.02).load(nacl)
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(zero_shift=(-0.01, 0.01)),
        profile=PeakProfileArtifacts(
            model="pseudo_voigt", fwhm=(0.04, 0.08), eta=(0.2, 0.8)
        ),
        intensity=IntensityArtifacts(peak_jitter=(0.9, 1.1)),
        background=BackgroundArtifacts(
            constant=(0.0, 0.02),
            amorphous_humps=(AmorphousHump(3.0, 0.8, 0.1),),
        ),
        noise=NoiseArtifacts(gaussian_std=(0.001, 0.01)),
        normalize_signal=True,
        domain="q",
        seed=17,
    )
    grid, first = calculator.pattern(domain="q", artifacts=artifacts)
    _, second = calculator.pattern(domain="q", artifacts=artifacts)
    np.testing.assert_array_equal(first, second)
    assert first.shape == grid.shape
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0)


def test_tch_profile_combines_instrument_size_and_microstrain(nacl):
    calculator = BraggCalculator(
        two_theta_range=(20.0, 80.0), two_theta_step=0.01
    ).load(nacl)
    instrument = PeakProfileArtifacts(
        model="tch",
        caglioti_u=0.002,
        caglioti_v=0.0,
        caglioti_w=0.003,
        lorentzian_x=0.002,
        lorentzian_y=0.003,
    )
    sample = PeakProfileArtifacts(
        model="tch",
        caglioti_u=0.002,
        caglioti_v=0.0,
        caglioti_w=0.003,
        lorentzian_x=0.002,
        lorentzian_y=0.003,
        crystallite_size_nm=20.0,
        microstrain=0.002,
    )
    _, instrument_pattern = calculator.pattern(
        artifacts=SimulationArtifacts(profile=instrument)
    )
    _, sample_pattern = calculator.pattern(artifacts=SimulationArtifacts(profile=sample))
    assert np.max(sample_pattern) < np.max(instrument_pattern)
    assert np.count_nonzero(sample_pattern > 0.01 * np.max(sample_pattern)) > np.count_nonzero(
        instrument_pattern > 0.01 * np.max(instrument_pattern)
    )


def test_pseudo_voigt_profile_preserves_integrated_peak_area():
    backend = NumpyBackend()
    grid = np.linspace(-10.0, 10.0, 40001)
    profile = _render_split_pseudo_voigt(
        grid,
        np.array([0.0]),
        np.array([3.0]),
        np.array([0.2]),
        np.array([0.2]),
        np.array([0.35]),
        backend,
        4_194_304,
    )
    assert np.trapezoid(profile, grid) == pytest.approx(3.0, rel=0.01)


def test_scherrer_width_is_converted_consistently_to_q(nacl):
    calculator = BraggCalculator().load(nacl)
    profile = PeakProfileArtifacts(
        model="tch",
        caglioti_w=1e-12,
        crystallite_size_nm=10.0,
        scherrer_constant=0.9,
    )
    width, eta = profile._tch_widths(
        calculator,
        "q",
        np.array([2.0]),
        calculator.backend,
        np.random.default_rng(1),
    )
    expected = 2.0 * np.pi * 0.9 / 100.0
    assert width[0] == pytest.approx(expected, rel=2e-4)
    assert eta[0] == pytest.approx(1.0, rel=2e-4)


def test_specimen_displacement_uses_bragg_brentano_geometry():
    backend = NumpyBackend()
    centers = np.array([20.0, 80.0])
    calibration = CalibrationArtifacts(
        specimen_displacement_mm=-0.08, goniometer_radius_mm=200.0
    )
    actual = calibration.apply(
        centers, "two_theta", backend, np.random.default_rng(1)
    )
    expected = centers + np.degrees(
        0.0008 * np.cos(np.radians(centers) / 2.0)
    )
    np.testing.assert_allclose(actual, expected)


def test_preferred_orientation_has_random_powder_limit(nacl):
    calculator = BraggCalculator().load(nacl)
    indices = calculator._domain_indices("two_theta")
    hkl = calculator._hkl["hkl"][indices]
    lattice = calculator._symm["lattice"]
    unity = PreferredOrientation(axis=(1, 0, 0), ratio=1.0).factors(
        hkl, lattice, calculator.backend, np.random.default_rng(1)
    )
    textured = PreferredOrientation(axis=(1, 0, 0), ratio=0.6).factors(
        hkl, lattice, calculator.backend, np.random.default_rng(1)
    )
    np.testing.assert_allclose(unity, 1.0)
    assert np.ptp(textured) > 0


def test_background_xy_file_is_interpolated_with_provenance(tmp_path, nacl):
    path = tmp_path / "empty_holder.xy"
    path.write_text("# 2theta intensity\n10,1\n45,3\n80,2\n")
    measured = BackgroundPattern.from_file(
        path, domain="two_theta", source="laboratory blank, instrument A"
    )
    calculator = BraggCalculator(
        two_theta_range=(10.0, 80.0), two_theta_step=35.0
    ).load(nacl)
    grid, ideal = calculator.pattern()
    artifacts = SimulationArtifacts(
        background=BackgroundArtifacts(measured=measured, measured_scale=2.0),
        clip_nonnegative=False,
    )
    _, actual = calculator.pattern(artifacts=artifacts)
    np.testing.assert_allclose(actual, ideal + np.array([2.0, 6.0, 4.0]))
    assert measured.source == "laboratory blank, instrument A"
    assert measured.source_sha256 == sha256(path.read_bytes()).hexdigest()
    assert not measured.coordinate.flags.writeable


def test_background_xye_uncertainty_and_weight_loading(tmp_path):
    sigma_path = tmp_path / "blank.xye"
    sigma_path.write_text("0 10 2\n1 12 3\n")
    sigma = BackgroundPattern.from_file(sigma_path, domain="q")
    np.testing.assert_allclose(sigma.uncertainty, [2, 3])

    weight_path = tmp_path / "blank.xy"
    weight_path.write_text("0 10 0.25\n1 12 0.04\n")
    weighted = BackgroundPattern.from_file(
        weight_path, domain="q", third_column="weight"
    )
    np.testing.assert_allclose(weighted.uncertainty, [2, 5])


def test_background_domain_and_coverage_are_explicit(tmp_path, nacl):
    path = tmp_path / "short.xy"
    path.write_text("1 2\n2 3\n")
    measured = BackgroundPattern.from_file(
        path, domain="q", third_column="ignore"
    )
    calculator = BraggCalculator(q_range=(0.0, 3.0), q_step=1.0).load(nacl)
    with pytest.raises(ValueError, match="do not cover"):
        calculator.pattern(
            domain="q",
            artifacts=SimulationArtifacts(
                background=BackgroundArtifacts(measured=measured)
            ),
        )
    with pytest.raises(ValueError, match="background uses domain"):
        calculator.pattern(
            artifacts=SimulationArtifacts(
                background=BackgroundArtifacts(
                    measured=measured, extrapolation="zero"
                )
            ),
        )


def test_background_library_validates_source_and_checksum(tmp_path):
    trace = tmp_path / "capillary.xye"
    trace.write_text("0 1 0.1\n1 2 0.1\n")
    digest = sha256(trace.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backgrounds": {
                    "empty-capillary": {
                        "path": trace.name,
                        "domain": "q",
                        "third_column": "sigma",
                        "source": "facility blank measurement DOI:example",
                        "sha256": digest,
                    }
                },
            }
        )
    )
    library = BackgroundLibrary(manifest)
    assert library.names == ("empty-capillary",)
    pattern = library.load("empty-capillary")
    np.testing.assert_allclose(pattern.intensity, [1, 2])
    assert pattern.source == "facility blank measurement DOI:example"

    trace.write_text("0 1 0.1\n1 3 0.1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        library.load("empty-capillary")


def test_background_library_requires_nonempty_source(tmp_path):
    trace = tmp_path / "blank.xy"
    trace.write_text("0 1\n1 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backgrounds": {
                    "unsourced": {
                        "path": trace.name,
                        "domain": "q",
                        "third_column": "ignore",
                        "source": "",
                        "sha256": sha256(trace.read_bytes()).hexdigest(),
                    }
                },
            }
        )
    )
    library = BackgroundLibrary(manifest)
    with pytest.raises(ValueError, match="non-empty source"):
        library.load("unsourced")


def test_bundled_background_library_is_valid_and_does_not_invent_data():
    library = BackgroundLibrary.bundled()
    assert library.names == ()


def test_amorphous_hump_has_requested_peak_height(nacl):
    calculator = BraggCalculator(q_range=(0.0, 4.0), q_step=0.01).load(nacl)
    grid, ideal = calculator.pattern(domain="q")
    artifacts = SimulationArtifacts(
        background=BackgroundArtifacts(
            amorphous_humps=(AmorphousHump(center=2.0, fwhm=0.5, height=4.0),)
        ),
        clip_nonnegative=False,
    )
    _, actual = calculator.pattern(domain="q", artifacts=artifacts)
    background = actual - ideal
    assert background[np.argmin(np.abs(grid - 2.0))] == pytest.approx(4.0)


def test_poisson_noise_returns_count_quantized_intensity(nacl):
    count_scale = 25.0
    calculator = BraggCalculator(q_range=(0.0, 2.0), q_step=0.02).load(nacl)
    _, observed = calculator.pattern(
        domain="q",
        artifacts=SimulationArtifacts(
            background=BackgroundArtifacts(constant=2.0),
            noise=NoiseArtifacts(poisson_count_scale=count_scale),
            seed=5,
        ),
    )
    np.testing.assert_allclose(observed * count_scale, np.round(observed * count_scale))


def test_correlated_noise_and_detector_effects(nacl):
    calculator = BraggCalculator(q_range=(0.0, 2.0), q_step=0.01).load(nacl)
    grid, observed = calculator.pattern(
        domain="q",
        artifacts=SimulationArtifacts(
            background=BackgroundArtifacts(constant=0.5),
            noise=NoiseArtifacts(correlated_std=0.1, correlation_length=0.05),
            detector=DetectorArtifacts(
                excluded_ranges=((0.5, 0.7),),
                saturation_level=0.8,
                quantization_step=0.02,
            ),
            seed=11,
        ),
    )
    assert np.all(observed[(grid >= 0.5) & (grid <= 0.7)] == 0)
    assert np.max(observed) <= 0.8
    np.testing.assert_allclose(observed / 0.02, np.round(observed / 0.02))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CalibrationArtifacts(axis_scale=0.0),
        lambda: PreferredOrientation(axis=(0, 0, 0)),
        lambda: IntensityArtifacts(peak_dropout_probability=1.1),
        lambda: PeakProfileArtifacts(model="pseudo_voigt", fwhm=0.0),
        lambda: PeakProfileArtifacts(model="calculator", microstrain=0.01),
        lambda: BackgroundArtifacts(measured_scale=-1.0),
        lambda: NoiseArtifacts(poisson_count_scale=0.0),
        lambda: DetectorArtifacts(saturation_level=0.0),
        lambda: SpuriousPeakArtifacts(count=(-1, 2)),
        lambda: SimulationArtifacts(domain="d_spacing"),
        lambda: SimulationArtifacts(seed=1.5),
    ],
)
def test_invalid_artifact_configuration_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_artifacts_argument_is_type_checked(nacl):
    calculator = BraggCalculator().load(nacl)
    with pytest.raises(TypeError, match="SimulationArtifacts"):
        calculator.pattern(artifacts={"noise": 0.1})
    with pytest.raises(TypeError, match="background must be"):
        SimulationArtifacts(background={"constant": 1.0})


def test_seeded_numpy_torch_artifact_parity(nacl):
    torch = pytest.importorskip("torch")
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(zero_shift=0.01, axis_scale=1.005),
        profile=PeakProfileArtifacts(
            model="tch",
            caglioti_u=0.002,
            caglioti_w=0.003,
            lorentzian_x=0.002,
            lorentzian_y=0.003,
            crystallite_size_nm=40.0,
            microstrain=0.001,
        ),
        intensity=IntensityArtifacts(
            scale=0.9,
            peak_jitter=(0.9, 1.1),
            peak_dropout_probability=0.1,
            preferred_orientation=PreferredOrientation((1, 0, 0), ratio=0.8),
        ),
        background=BackgroundArtifacts(
            constant=0.01,
            linear_slope=0.001,
            amorphous_humps=(AmorphousHump(3.0, 0.7, 0.05),),
        ),
        noise=NoiseArtifacts(gaussian_std=0.005),
        detector=DetectorArtifacts(
            random_mask_probability=0.05, saturation_level=1.0
        ),
        spurious_peaks=SpuriousPeakArtifacts(
            count=2, intensity=(0.02, 0.04), fwhm=0.06, eta=0.4
        ),
        normalize_signal=True,
        final_normalize=True,
        domain="q",
        seed=42,
    )
    numpy_calc = BraggCalculator(backend=NumpyBackend(), q_step=0.02).load(nacl)
    torch_calc = BraggCalculator(backend=TorchBackend(), q_step=0.02).load(nacl)
    nx, ny = numpy_calc.pattern(domain="q", artifacts=artifacts)
    tx, ty = torch_calc.pattern(domain="q", artifacts=artifacts)
    np.testing.assert_allclose(tx.cpu(), nx, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(ty.cpu(), ny, rtol=2e-6, atol=1e-9)
    assert torch.all(torch.isfinite(ty))


def test_torch_gradient_survives_continuous_artifacts(triclinic_structure):
    torch = pytest.importorskip("torch")
    calculator = BraggCalculator(
        backend=TorchBackend(), q_step=0.02
    ).load(triclinic_structure)
    parameters = calculator.tensor_parameters(requires_grad=["frac_coords"])
    _, pattern = calculator.pattern(
        domain="q",
        parameters=parameters,
        artifacts=SimulationArtifacts(
            profile=PeakProfileArtifacts(
                model="tch", caglioti_u=0.002, caglioti_w=0.004
            ),
            background=BackgroundArtifacts(constant=0.1),
            noise=NoiseArtifacts(gaussian_std=0.01),
            seed=3,
        ),
    )
    pattern.sum().backward()
    gradient = parameters["frac_coords"].grad
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient))
