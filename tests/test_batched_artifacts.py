import inspect

import pytest

from braggcalculator import (
    AmorphousHump,
    BackgroundArtifacts,
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
    apply_peak_artifact_batch,
    render_artifact_batch,
)


torch = pytest.importorskip("torch")


def _lines(*, batch=2, dtype=torch.float32, device="cpu"):
    positions = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0], [1.2, 2.3, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )[:batch]
    intensities = torch.tensor(
        [[1.0, 0.5, 0.2, 0.0], [0.8, 0.4, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )[:batch]
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]],
        device=device,
    )[:batch]
    return positions, intensities, mask


def test_peak_batch_applies_only_position_and_intensity_effects():
    positions, intensities, mask = _lines()
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(
            zero_shift=0.1, axis_scale=2.0, peak_jitter_std=0.0
        ),
        intensity=IntensityArtifacts(scale=3.0, peak_jitter=1.0),
        background=BackgroundArtifacts(constant=100.0),
        noise=NoiseArtifacts(gaussian_std=100.0),
        spurious_peaks=SpuriousPeakArtifacts(count=3),
        seed=4,
    )
    moved, scaled, actual_mask = apply_peak_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        artifacts=artifacts,
    )
    torch.testing.assert_close(moved[mask], (positions * 2.0 + 0.1)[mask])
    torch.testing.assert_close(scaled, intensities * mask * 3.0)
    torch.testing.assert_close(actual_mask, mask)


def test_peak_batch_seed_is_repeatable_and_padding_is_explicit():
    positions, intensities, mask = _lines()
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(peak_jitter_std=0.02),
        intensity=IntensityArtifacts(
            peak_jitter=(0.8, 1.2), peak_dropout_probability=0.2
        ),
        seed=81,
    )
    first = apply_peak_artifact_batch(
        positions, intensities, peak_mask=mask, artifacts=artifacts
    )
    second = apply_peak_artifact_batch(
        positions, intensities, peak_mask=mask, artifacts=artifacts
    )
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.all(first[1][~first[2]] == 0)


def test_single_pattern_peak_position_jitter_is_reproducible(nacl):
    calculator = BraggCalculator(q_step=0.02).load(nacl)
    baseline_grid, baseline = calculator.pattern(domain="q")
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(peak_jitter_std=0.01),
        seed=12,
    )
    first_grid, first = calculator.pattern(domain="q", artifacts=artifacts)
    second_grid, second = calculator.pattern(domain="q", artifacts=artifacts)
    torch.testing.assert_close(
        torch.as_tensor(first_grid), torch.as_tensor(second_grid)
    )
    torch.testing.assert_close(torch.as_tensor(first), torch.as_tensor(second))
    torch.testing.assert_close(
        torch.as_tensor(first_grid), torch.as_tensor(baseline_grid)
    )
    assert not torch.equal(torch.as_tensor(first), torch.as_tensor(baseline))


def test_dense_pseudo_voigt_batch_preserves_shape_device_dtype_and_area():
    positions = torch.tensor([[2.0], [2.5]], dtype=torch.float64)
    intensities = torch.tensor([[3.0], [2.0]], dtype=torch.float64)
    grid = torch.linspace(-10.0, 10.0, 40001, dtype=torch.float64)
    artifacts = SimulationArtifacts(
        profile=PeakProfileArtifacts(model="pseudo_voigt", fwhm=0.2, eta=0.35),
        clip_nonnegative=False,
    )
    patterns = render_artifact_batch(
        positions, intensities, grid=grid, artifacts=artifacts
    )
    assert patterns.shape == (2, grid.numel())
    assert patterns.device == positions.device
    assert patterns.dtype == positions.dtype
    areas = torch.trapezoid(patterns, grid, dim=-1)
    torch.testing.assert_close(areas, intensities[:, 0], rtol=0.01, atol=0.0)


def test_dense_batch_supports_calculator_profile_parameters():
    positions, intensities, mask = _lines()
    grid = torch.linspace(0.0, 4.0, 401)
    artifacts = SimulationArtifacts(clip_nonnegative=False)
    with pytest.raises(ValueError, match="profile_fwhm"):
        render_artifact_batch(
            positions,
            intensities,
            peak_mask=mask,
            grid=grid,
            artifacts=artifacts,
        )
    patterns = render_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        grid=grid,
        artifacts=artifacts,
        profile_fwhm=torch.tensor([0.08, 0.12]),
    )
    assert patterns.shape == (2, 401)
    assert torch.all(torch.isfinite(patterns))


def test_generator_and_configuration_seed_are_mutually_exclusive():
    positions, intensities, mask = _lines()
    artifacts = SimulationArtifacts(seed=2)
    generator = torch.Generator().manual_seed(3)
    with pytest.raises(ValueError, match="either artifacts.seed or generator"):
        apply_peak_artifact_batch(
            positions,
            intensities,
            peak_mask=mask,
            artifacts=artifacts,
            generator=generator,
        )


def test_preferred_orientation_requires_crystallographic_metadata():
    positions, intensities, mask = _lines()
    artifacts = SimulationArtifacts(
        intensity=IntensityArtifacts(
            preferred_orientation=PreferredOrientation(axis=(1, 0, 0), ratio=0.8)
        )
    )
    with pytest.raises(ValueError, match="requires hkl"):
        apply_peak_artifact_batch(
            positions, intensities, peak_mask=mask, artifacts=artifacts
        )


def test_full_dense_pipeline_covers_phase_two_artifacts():
    positions, intensities, mask = _lines(dtype=torch.float64)
    grid = torch.linspace(0.2, 4.0, 512, dtype=torch.float64)
    hkl = torch.tensor(
        [
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )
    lattice = torch.eye(3, dtype=torch.float64).expand(2, -1, -1).clone()
    measured = torch.linspace(0.01, 0.03, grid.numel(), dtype=torch.float64)
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(
            zero_shift=(-0.005, 0.005), axis_scale=(0.998, 1.002)
        ),
        intensity=IntensityArtifacts(
            scale=(0.8, 1.2),
            peak_jitter=(0.9, 1.1),
            peak_dropout_probability=0.1,
            preferred_orientation=PreferredOrientation(
                axis=(1, 0, 0), ratio=(0.8, 1.2), fraction=(0.5, 1.0)
            ),
        ),
        profile=PeakProfileArtifacts(
            model="tch",
            caglioti_u=(0.001, 0.003),
            caglioti_w=(0.002, 0.005),
            lorentzian_x=(0.001, 0.003),
            lorentzian_y=(0.001, 0.003),
            crystallite_size_nm=(20.0, 80.0),
            microstrain=(0.0001, 0.001),
        ),
        background=BackgroundArtifacts(
            constant=(0.0, 0.02),
            linear_slope=(-0.002, 0.002),
            chebyshev_coefficients=(0.01, 0.002, -0.001),
            amorphous_humps=(
                AmorphousHump(
                    center=(1.5, 2.0),
                    fwhm=(0.4, 0.8),
                    height=(0.02, 0.08),
                    eta=(0.0, 0.3),
                ),
            ),
            measured_scale=(0.5, 1.5),
            measured_offset=(0.0, 0.01),
        ),
        spurious_peaks=SpuriousPeakArtifacts(
            count=(1, 3),
            intensity=(0.01, 0.05),
            fwhm=(0.03, 0.08),
            eta=(0.2, 0.8),
        ),
        noise=NoiseArtifacts(
            gaussian_std=(0.001, 0.003),
            correlated_std=(0.001, 0.004),
            correlation_length=(0.02, 0.08),
            poisson_count_scale=(5000.0, 10000.0),
        ),
        detector=DetectorArtifacts(
            random_mask_probability=0.01,
            excluded_ranges=((3.6, 3.7),),
            saturation_level=2.0,
            quantization_step=0.001,
        ),
        normalize_signal=True,
        final_normalize=True,
        domain="q",
        seed=25,
    )
    first = render_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        grid=grid,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        wavelength=torch.tensor([1.5406, 1.5406], dtype=torch.float64),
        measured_background=measured,
        max_entries=4096,
    )
    second = render_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        grid=grid,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        wavelength=1.5406,
        measured_background=measured,
        max_entries=4096,
    )
    torch.testing.assert_close(first, second)
    assert first.shape == (2, grid.numel())
    assert torch.all(torch.isfinite(first))
    assert torch.all(first >= 0)
    assert torch.all(first <= 1)
    assert torch.all(first[:, (grid >= 3.6) & (grid <= 3.7)] == 0)


def test_two_theta_batch_supports_displacement_and_asymmetry():
    positions = torch.tensor([[20.0, 50.0]], dtype=torch.float64)
    intensities = torch.ones_like(positions)
    grid = torch.linspace(10.0, 70.0, 1001, dtype=torch.float64)
    artifacts = SimulationArtifacts(
        calibration=CalibrationArtifacts(specimen_displacement_mm=-0.08),
        profile=PeakProfileArtifacts(
            model="pseudo_voigt",
            fwhm=0.1,
            eta=0.3,
            axial_asymmetry=0.05,
        ),
        domain="two_theta",
        clip_nonnegative=False,
    )
    pattern = render_artifact_batch(
        positions,
        intensities,
        grid=grid,
        artifacts=artifacts,
        domain="two_theta",
    )
    assert pattern.shape == (1, 1001)
    assert torch.all(torch.isfinite(pattern))


def test_measured_background_object_is_interpolated_on_device(tmp_path):
    path = tmp_path / "blank.xy"
    path.write_text("0 1\n2 3\n4 1\n")
    measured = BackgroundPattern.from_file(
        path, domain="q", third_column="ignore"
    )
    positions = torch.empty((1, 0), dtype=torch.float64)
    intensities = torch.empty_like(positions)
    grid = torch.linspace(0.0, 4.0, 5, dtype=torch.float64)
    artifacts = SimulationArtifacts(
        profile=PeakProfileArtifacts(model="pseudo_voigt"),
        background=BackgroundArtifacts(measured=measured),
        clip_nonnegative=False,
    )
    pattern = render_artifact_batch(
        positions, intensities, grid=grid, artifacts=artifacts
    )
    torch.testing.assert_close(
        pattern, torch.tensor([[1.0, 2.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
    )


def test_batched_artifact_source_has_no_numpy_or_cpu_round_trip():
    import braggcalculator.batched_artifacts as module

    source = inspect.getsource(module)
    assert ".numpy(" not in source
    assert ".cpu(" not in source
    assert ".item(" not in source


def test_dense_batch_retains_gradients_for_continuous_pipeline():
    positions, intensities, mask = _lines(dtype=torch.float64)
    positions.requires_grad_()
    intensities.requires_grad_()
    grid = torch.linspace(0.0, 4.0, 301, dtype=torch.float64)
    artifacts = SimulationArtifacts(
        profile=PeakProfileArtifacts(model="pseudo_voigt", fwhm=0.1, eta=0.4),
        background=BackgroundArtifacts(constant=0.1),
        clip_nonnegative=False,
    )
    pattern = render_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        grid=grid,
        artifacts=artifacts,
    )
    pattern.square().sum().backward()
    assert positions.grad is not None
    assert intensities.grad is not None
    assert torch.all(torch.isfinite(positions.grad))
    assert torch.all(torch.isfinite(intensities.grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dense_batch_stays_on_cuda():
    positions, intensities, mask = _lines(device="cuda")
    grid = torch.linspace(0.0, 4.0, 401, device="cuda")
    artifacts = SimulationArtifacts(
        profile=PeakProfileArtifacts(model="pseudo_voigt", fwhm=(0.05, 0.1)),
        noise=NoiseArtifacts(
            gaussian_std=0.01,
            correlated_std=0.01,
            poisson_count_scale=1000.0,
        ),
        seed=7,
    )
    pattern = render_artifact_batch(
        positions,
        intensities,
        peak_mask=mask,
        grid=grid,
        artifacts=artifacts,
    )
    assert pattern.is_cuda
    assert pattern.dtype == torch.float32
