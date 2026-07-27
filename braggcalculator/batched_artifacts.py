"""Stateless, device-native artifact simulation for batches of powder lines.

This module deliberately imports PyTorch lazily so the base BraggCalculator
installation remains usable without the optional ``torch`` extra.
"""

from __future__ import annotations

from math import log, pi, sqrt

from .artifacts import (
    Domain,
    IntegerRange,
    ScalarRange,
    SimulationArtifacts,
    _bounds,
    _integer_bounds,
)


_GAUSSIAN_INTEGRAL = sqrt(pi) / (4.0 * sqrt(log(2.0)))
_FWHM_TO_SIGMA = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "batched artifact simulation requires the optional PyTorch dependency; "
            "install braggcalculator[torch]"
        ) from exc
    return torch


def _sample(
    torch,
    value: ScalarRange,
    shape: tuple[int, ...],
    reference,
    generator,
):
    lower, upper = _bounds(value)
    if lower == upper:
        return torch.full(shape, lower, dtype=reference.dtype, device=reference.device)
    return lower + (upper - lower) * torch.rand(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _sample_integer(
    torch,
    value: IntegerRange,
    shape: tuple[int, ...],
    reference,
    generator,
):
    lower, upper = _integer_bounds(value)
    if lower == upper:
        return torch.full(shape, lower, dtype=torch.long, device=reference.device)
    return torch.randint(
        lower,
        upper + 1,
        shape,
        dtype=torch.long,
        device=reference.device,
        generator=generator,
    )


def _resolve_generator(torch, reference, artifacts, generator):
    if generator is not None and artifacts.seed is not None:
        raise ValueError("pass either artifacts.seed or generator, not both")
    if generator is not None or artifacts.seed is None:
        return generator
    seeded = torch.Generator(device=reference.device)
    seeded.manual_seed(artifacts.seed)
    return seeded


def _validate_inputs(
    torch,
    positions,
    intensities,
    peak_mask,
    *,
    domain,
    artifacts,
    hkl,
    lattice,
    wavelength,
):
    if not isinstance(artifacts, SimulationArtifacts):
        raise TypeError("artifacts must be a SimulationArtifacts instance")
    if domain not in {"q", "two_theta"}:
        raise ValueError("domain must be 'q' or 'two_theta'")
    if artifacts.domain is not None and artifacts.domain != domain:
        raise ValueError(
            f"this artifact configuration requires domain={artifacts.domain!r}, "
            f"got {domain!r}"
        )
    if not isinstance(positions, torch.Tensor) or not isinstance(intensities, torch.Tensor):
        raise TypeError("positions and intensities must be torch.Tensor objects")
    if positions.ndim != 2 or intensities.shape != positions.shape:
        raise ValueError("positions and intensities must have equal [batch, peaks] shapes")
    if not positions.is_floating_point() or not intensities.is_floating_point():
        raise TypeError("positions and intensities must use floating-point dtypes")
    if positions.device != intensities.device or positions.dtype != intensities.dtype:
        raise ValueError("positions and intensities must have the same device and dtype")

    if peak_mask is None:
        peak_mask = torch.ones_like(positions, dtype=torch.bool)
    elif not isinstance(peak_mask, torch.Tensor):
        raise TypeError("peak_mask must be a torch.Tensor or None")
    elif peak_mask.shape != positions.shape or peak_mask.dtype != torch.bool:
        raise ValueError("peak_mask must be boolean with shape [batch, peaks]")
    elif peak_mask.device != positions.device:
        raise ValueError("peak_mask must be on the same device as positions")

    batch_size, peak_count = positions.shape
    if hkl is not None:
        if not isinstance(hkl, torch.Tensor) or hkl.shape != (batch_size, peak_count, 3):
            raise ValueError("hkl must have shape [batch, peaks, 3]")
        if hkl.device != positions.device:
            raise ValueError("hkl must be on the same device as positions")
    if lattice is not None:
        if not isinstance(lattice, torch.Tensor) or lattice.shape != (batch_size, 3, 3):
            raise ValueError("lattice must have shape [batch, 3, 3]")
        if lattice.device != positions.device:
            raise ValueError("lattice must be on the same device as positions")

    if wavelength is not None:
        wavelength = torch.as_tensor(
            wavelength, dtype=positions.dtype, device=positions.device
        )
        if wavelength.ndim == 0:
            wavelength = wavelength.expand(batch_size)
        elif wavelength.shape != (batch_size,):
            raise ValueError("wavelength must be a scalar or have shape [batch]")
    return peak_mask, wavelength


def _apply_peak_effects(
    torch,
    positions,
    intensities,
    peak_mask,
    *,
    domain,
    artifacts,
    hkl,
    lattice,
    generator,
):
    batch_size, peak_count = positions.shape
    calibration = artifacts.calibration
    scale = _sample(torch, calibration.axis_scale, (batch_size, 1), positions, generator)
    zero_shift = _sample(
        torch, calibration.zero_shift, (batch_size, 1), positions, generator
    )
    transformed_positions = positions * scale + zero_shift

    jitter_std = _sample(
        torch, calibration.peak_jitter_std, (batch_size, 1), positions, generator
    )
    if _bounds(calibration.peak_jitter_std)[1] > 0:
        transformed_positions = transformed_positions + jitter_std * torch.randn(
            (batch_size, peak_count),
            dtype=positions.dtype,
            device=positions.device,
            generator=generator,
        )

    if _bounds(calibration.specimen_displacement_mm) != (0.0, 0.0):
        if domain != "two_theta":
            raise ValueError(
                "specimen displacement is only defined for domain='two_theta'"
            )
        displacement = _sample(
            torch,
            calibration.specimen_displacement_mm,
            (batch_size, 1),
            positions,
            generator,
        )
        theta = positions * scale * (torch.pi / 180.0) / 2.0
        shift_radians = (
            -2.0
            * displacement
            * torch.cos(theta)
            / calibration.goniometer_radius_mm
        )
        transformed_positions = transformed_positions + torch.rad2deg(shift_radians)

    intensity_config = artifacts.intensity
    intensity_scale = _sample(
        torch, intensity_config.scale, (batch_size, 1), positions, generator
    )
    transformed_intensities = intensities * intensity_scale
    jitter = _sample(
        torch,
        intensity_config.peak_jitter,
        (batch_size, peak_count),
        positions,
        generator,
    )
    transformed_intensities = transformed_intensities * jitter
    if intensity_config.peak_dropout_probability:
        keep = (
            torch.rand(
                (batch_size, peak_count),
                dtype=positions.dtype,
                device=positions.device,
                generator=generator,
            )
            >= intensity_config.peak_dropout_probability
        )
        peak_mask = peak_mask & keep

    orientation = intensity_config.preferred_orientation
    if orientation is not None:
        if hkl is None or lattice is None:
            raise ValueError(
                "preferred orientation requires hkl [batch, peaks, 3] and "
                "lattice [batch, 3, 3]"
            )
        lattice_values = lattice.to(dtype=positions.dtype)
        reciprocal_metric = torch.linalg.inv(
            lattice_values @ lattice_values.transpose(-1, -2)
        )
        hkl_values = hkl.to(dtype=positions.dtype)
        axis = torch.tensor(
            orientation.axis, dtype=positions.dtype, device=positions.device
        ).expand(batch_size, -1)
        numerator = torch.einsum(
            "bpi,bij,bj->bp", hkl_values, reciprocal_metric, axis
        )
        hkl_norm2 = torch.einsum(
            "bpi,bij,bpj->bp", hkl_values, reciprocal_metric, hkl_values
        )
        axis_norm2 = torch.einsum(
            "bi,bij,bj->b", axis, reciprocal_metric, axis
        ).unsqueeze(-1)
        tiny = torch.finfo(positions.dtype).tiny
        cos2 = torch.clamp(
            numerator.square() / torch.clamp(hkl_norm2 * axis_norm2, min=tiny),
            0.0,
            1.0,
        )
        ratio = _sample(
            torch, orientation.ratio, (batch_size, 1), positions, generator
        )
        fraction = _sample(
            torch, orientation.fraction, (batch_size, 1), positions, generator
        )
        march = (
            ratio.square() * cos2 + (1.0 - cos2) / ratio
        ).pow(-1.5)
        transformed_intensities = transformed_intensities * (
            (1.0 - fraction) + fraction * march
        )

    transformed_intensities = torch.where(
        peak_mask, transformed_intensities, torch.zeros_like(transformed_intensities)
    )
    return transformed_positions, transformed_intensities, peak_mask


def apply_peak_artifact_batch(
    positions,
    intensities,
    *,
    artifacts: SimulationArtifacts,
    peak_mask=None,
    domain: Domain = "q",
    hkl=None,
    lattice=None,
    wavelength=None,
    generator=None,
):
    """Apply only peak-position and peak-intensity effects to a Torch batch.

    The returned tuple is ``(positions, intensities, peak_mask)``. It is the
    appropriate representation for models that consume discrete powder lines.
    Backgrounds, profile rendering, noise, detector effects, and spurious
    peaks are intentionally not included; use :func:`render_artifact_batch`
    for those effects.
    """

    torch = _torch()
    peak_mask, _ = _validate_inputs(
        torch,
        positions,
        intensities,
        peak_mask,
        domain=domain,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        wavelength=wavelength,
    )
    generator = _resolve_generator(torch, positions, artifacts, generator)
    return _apply_peak_effects(
        torch,
        positions,
        intensities,
        peak_mask,
        domain=domain,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        generator=generator,
    )


def _grid_batch(torch, grid, positions):
    if not isinstance(grid, torch.Tensor) or not grid.is_floating_point():
        raise TypeError("grid must be a floating-point torch.Tensor")
    if grid.device != positions.device or grid.dtype != positions.dtype:
        raise ValueError("grid must have the same device and dtype as positions")
    batch_size = positions.shape[0]
    if grid.ndim == 1:
        if grid.shape[0] < 2:
            raise ValueError("grid must contain at least two points")
        return grid.unsqueeze(0).expand(batch_size, -1)
    if grid.ndim == 2 and grid.shape[0] == batch_size and grid.shape[1] >= 2:
        return grid
    raise ValueError("grid must have shape [grid] or [batch, grid]")


def _broadcast_peak_parameter(torch, value, positions, name):
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=positions.dtype, device=positions.device)
    if tensor.ndim == 0:
        return tensor.expand_as(positions)
    if tensor.shape == (positions.shape[0],):
        return tensor.unsqueeze(-1).expand_as(positions)
    if tensor.shape == positions.shape:
        return tensor
    raise ValueError(
        f"{name} must be scalar, [batch], or [batch, peaks]"
    )


def _profile_parameters(
    torch,
    centers,
    *,
    domain,
    artifacts,
    wavelength,
    profile_fwhm,
    profile_eta,
    generator,
):
    batch_size, peak_count = centers.shape
    profile = artifacts.profile
    if profile.model == "calculator":
        fwhm = _broadcast_peak_parameter(
            torch, profile_fwhm, centers, "profile_fwhm"
        )
        if fwhm is None:
            raise ValueError(
                "batched rendering with profile.model='calculator' requires "
                "profile_fwhm; alternatively select model='pseudo_voigt' or 'tch'"
            )
        eta = _broadcast_peak_parameter(torch, profile_eta, centers, "profile_eta")
        if eta is None:
            eta = torch.zeros_like(centers)
    elif profile.model == "pseudo_voigt":
        fwhm = _sample(
            torch, profile.fwhm, (batch_size, 1), centers, generator
        ).expand(batch_size, peak_count)
        eta = _sample(
            torch, profile.eta, (batch_size, 1), centers, generator
        ).expand(batch_size, peak_count)
    else:
        if domain == "q":
            if wavelength is None:
                raise ValueError("a wavelength is required for a q-domain TCH profile")
            argument = torch.clamp(
                centers * wavelength.unsqueeze(-1) / (4.0 * torch.pi),
                -1.0,
                1.0,
            )
            theta = torch.arcsin(argument)
        else:
            theta = torch.deg2rad(centers) / 2.0
        tangent = torch.tan(theta)
        u = _sample(
            torch, profile.caglioti_u, (batch_size, 1), centers, generator
        )
        v = _sample(
            torch, profile.caglioti_v, (batch_size, 1), centers, generator
        )
        w = _sample(
            torch, profile.caglioti_w, (batch_size, 1), centers, generator
        )
        gaussian = torch.sqrt(u * tangent.square() + v * tangent + w)
        x = _sample(
            torch, profile.lorentzian_x, (batch_size, 1), centers, generator
        )
        y = _sample(
            torch, profile.lorentzian_y, (batch_size, 1), centers, generator
        )
        lorentzian = x / torch.cos(theta) + y * tangent

        if profile.crystallite_size_nm is not None:
            if wavelength is None:
                raise ValueError("crystallite-size broadening requires wavelength")
            size_nm = _sample(
                torch,
                profile.crystallite_size_nm,
                (batch_size, 1),
                centers,
                generator,
            )
            size_radians = (
                profile.scherrer_constant
                * wavelength.unsqueeze(-1)
                / (10.0 * size_nm * torch.cos(theta))
            )
            lorentzian = lorentzian + torch.rad2deg(size_radians)
        microstrain = _sample(
            torch, profile.microstrain, (batch_size, 1), centers, generator
        )
        lorentzian = torch.clamp(
            lorentzian + torch.rad2deg(4.0 * microstrain * tangent), min=0.0
        )
        combined_fifth = (
            gaussian**5
            + 2.69269 * gaussian**4 * lorentzian
            + 2.42843 * gaussian**3 * lorentzian**2
            + 4.47163 * gaussian**2 * lorentzian**3
            + 0.07842 * gaussian * lorentzian**4
            + lorentzian**5
        )
        fwhm_degrees = combined_fifth.pow(0.2)
        ratio = lorentzian / fwhm_degrees
        eta = torch.clamp(
            1.36603 * ratio - 0.47719 * ratio.square() + 0.11116 * ratio**3,
            0.0,
            1.0,
        )
        if domain == "q":
            dq_ddegree = (
                2.0
                * torch.pi
                * torch.cos(theta)
                / wavelength.unsqueeze(-1)
                * (torch.pi / 180.0)
            )
            fwhm = fwhm_degrees * dq_ddegree
        else:
            fwhm = fwhm_degrees

    asymmetry = _sample(
        torch, profile.axial_asymmetry, (batch_size, 1), centers, generator
    )
    if _bounds(profile.axial_asymmetry)[1] > 0:
        if domain != "two_theta":
            raise ValueError("axial_asymmetry is only defined for domain='two_theta'")
        tangent = torch.clamp(torch.tan(torch.deg2rad(centers) / 2.0), min=1e-8)
        low_fwhm = fwhm * (1.0 + asymmetry / tangent)
    else:
        low_fwhm = fwhm
    return low_fwhm, fwhm, eta


def _render_split_pseudo_voigt(
    torch,
    grid,
    centers,
    intensities,
    peak_mask,
    low_fwhm,
    high_fwhm,
    eta,
    *,
    max_entries,
):
    batch_size, peak_count = centers.shape
    grid_count = grid.shape[1]
    result = torch.zeros(
        (batch_size, grid_count), dtype=centers.dtype, device=centers.device
    )
    if peak_count == 0:
        return result
    peak_chunk = max(1, max_entries // max(batch_size * grid_count, 1))
    for start in range(0, peak_count, peak_chunk):
        stop = min(start + peak_chunk, peak_count)
        offset = grid[:, :, None] - centers[:, None, start:stop]
        low = low_fwhm[:, None, start:stop]
        high = high_fwhm[:, None, start:stop]
        width = torch.where(offset < 0.0, low, high)
        gaussian = torch.exp(-4.0 * log(2.0) * (offset / width).square())
        gaussian = gaussian / (_GAUSSIAN_INTEGRAL * (low + high))
        lorentzian = 1.0 / (1.0 + 4.0 * (offset / width).square())
        lorentzian = lorentzian / (0.25 * pi * (low + high))
        mixing = eta[:, None, start:stop]
        shape = (1.0 - mixing) * gaussian + mixing * lorentzian
        areas = intensities[:, None, start:stop] * peak_mask[
            :, None, start:stop
        ].to(dtype=centers.dtype)
        result = result + torch.sum(areas * shape, dim=-1)
    return result


def _normalize(torch, values):
    return values / (torch.amax(values, dim=-1, keepdim=True) + 1e-16)


def _interpolate_measured(torch, measured, grid, domain, extrapolation):
    if measured.domain != domain:
        raise ValueError(
            f"background uses domain={measured.domain!r}, but the pattern uses {domain!r}"
        )
    source_x = torch.tensor(
        measured.coordinate, dtype=grid.dtype, device=grid.device
    )
    source_y = torch.tensor(
        measured.intensity, dtype=grid.dtype, device=grid.device
    )
    indices = torch.searchsorted(source_x, grid.contiguous())
    below = grid < source_x[0]
    above = grid > source_x[-1]
    right = torch.clamp(indices, 1, source_x.shape[0] - 1)
    left = right - 1
    x0 = source_x[left]
    x1 = source_x[right]
    y0 = source_y[left]
    y1 = source_y[right]
    result = y0 + (grid - x0) * (y1 - y0) / (x1 - x0)
    outside = below | above
    if extrapolation == "zero":
        result = torch.where(outside, torch.zeros_like(result), result)
    elif extrapolation == "edge":
        result = torch.where(below, source_y[0], result)
        result = torch.where(above, source_y[-1], result)
    elif extrapolation == "error":
        torch._assert(
            torch.all(~outside),
            "background samples do not cover the simulation grid",
        )
    else:  # protected by BackgroundArtifacts validation
        raise ValueError("extrapolation must be 'error', 'zero', or 'edge'")
    return result


def _background(
    torch,
    grid,
    *,
    domain,
    artifacts,
    measured_background,
    generator,
):
    batch_size, grid_count = grid.shape
    config = artifacts.background
    result = torch.zeros_like(grid)
    center = 0.5 * (grid[:, :1] + grid[:, -1:])
    result = result + _sample(
        torch, config.constant, (batch_size, 1), grid, generator
    )
    result = result + _sample(
        torch, config.linear_slope, (batch_size, 1), grid, generator
    ) * (grid - center)

    if config.chebyshev_coefficients:
        span = grid[:, -1:] - grid[:, :1]
        normalized = torch.where(
            span != 0.0,
            2.0 * (grid - grid[:, :1]) / span - 1.0,
            torch.zeros_like(grid),
        )
        t0 = torch.ones_like(grid)
        result = result + config.chebyshev_coefficients[0] * t0
        if len(config.chebyshev_coefficients) > 1:
            t1 = normalized
            result = result + config.chebyshev_coefficients[1] * t1
            for coefficient in config.chebyshev_coefficients[2:]:
                t2 = 2.0 * normalized * t1 - t0
                result = result + coefficient * t2
                t0, t1 = t1, t2

    for hump in config.amorphous_humps:
        hump_center = _sample(
            torch, hump.center, (batch_size, 1), grid, generator
        )
        width = _sample(torch, hump.fwhm, (batch_size, 1), grid, generator)
        height = _sample(torch, hump.height, (batch_size, 1), grid, generator)
        eta = _sample(torch, hump.eta, (batch_size, 1), grid, generator)
        delta = grid - hump_center
        gaussian = torch.exp(
            -0.5 * (delta / (width * _FWHM_TO_SIGMA)).square()
        )
        lorentzian = 1.0 / (1.0 + (2.0 * delta / width).square())
        result = result + height * (
            (1.0 - eta) * gaussian + eta * lorentzian
        )

    measured = measured_background
    if measured is not None:
        if not isinstance(measured, torch.Tensor):
            raise TypeError("measured_background must be a torch.Tensor")
        if measured.device != grid.device or measured.dtype != grid.dtype:
            raise ValueError(
                "measured_background must have the same device and dtype as grid"
            )
        if measured.ndim == 1 and measured.shape[0] == grid_count:
            measured = measured.unsqueeze(0).expand(batch_size, -1)
        elif measured.shape != grid.shape:
            raise ValueError(
                "measured_background must have shape [grid] or [batch, grid]"
            )
    elif config.measured is not None:
        measured = _interpolate_measured(
            torch, config.measured, grid, domain, config.extrapolation
        )
    if measured is not None:
        measured_scale = _sample(
            torch, config.measured_scale, (batch_size, 1), grid, generator
        )
        measured_offset = _sample(
            torch, config.measured_offset, (batch_size, 1), grid, generator
        )
        result = result + measured_scale * measured + measured_offset
    return result


def _spurious_peaks(torch, grid, artifacts, generator, max_entries):
    batch_size, _ = grid.shape
    config = artifacts.spurious_peaks
    _, count_max = _integer_bounds(config.count)
    if count_max == 0:
        return torch.zeros_like(grid)
    counts = _sample_integer(
        torch, config.count, (batch_size,), grid, generator
    )
    mask = (
        torch.arange(count_max, device=grid.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )
    unit = torch.rand(
        (batch_size, count_max),
        dtype=grid.dtype,
        device=grid.device,
        generator=generator,
    )
    centers = grid[:, :1] + unit * (grid[:, -1:] - grid[:, :1])
    intensities = _sample(
        torch, config.intensity, (batch_size, count_max), grid, generator
    )
    widths = _sample(
        torch, config.fwhm, (batch_size, count_max), grid, generator
    )
    eta = _sample(
        torch, config.eta, (batch_size, count_max), grid, generator
    )
    return _render_split_pseudo_voigt(
        torch,
        grid,
        centers,
        intensities,
        mask,
        widths,
        widths,
        eta,
        max_entries=max_entries,
    )


def _noise(torch, values, grid, artifacts, generator):
    batch_size, grid_count = grid.shape
    config = artifacts.noise
    if config.poisson_count_scale is not None:
        count_scale = _sample(
            torch,
            config.poisson_count_scale,
            (batch_size, 1),
            grid,
            generator,
        )
        expected = torch.clamp(values, min=0.0) * count_scale
        values = torch.poisson(expected, generator=generator) / count_scale

    if _bounds(config.gaussian_std)[1] > 0:
        std = _sample(
            torch, config.gaussian_std, (batch_size, 1), grid, generator
        )
        values = values + std * torch.randn(
            (batch_size, grid_count),
            dtype=grid.dtype,
            device=grid.device,
            generator=generator,
        )

    if _bounds(config.correlated_std)[1] > 0:
        if grid_count < 2:
            raise ValueError("correlated noise requires at least two grid points")
        std = _sample(
            torch, config.correlated_std, (batch_size, 1), grid, generator
        )
        length = _sample(
            torch, config.correlation_length, (batch_size, 1), grid, generator
        )
        step = torch.mean(torch.abs(torch.diff(grid, dim=-1)), dim=-1, keepdim=True)
        sigma_points = length / step
        padded_count = 2 * grid_count
        white = torch.randn(
            (batch_size, padded_count),
            dtype=grid.dtype,
            device=grid.device,
            generator=generator,
        )
        spectrum = torch.fft.rfft(white, dim=-1)
        angular_frequency = (
            2.0
            * torch.pi
            * torch.fft.rfftfreq(
                padded_count, d=1.0, dtype=grid.dtype, device=grid.device
            )
        )
        gaussian_filter = torch.exp(
            -0.5 * (sigma_points * angular_frequency.unsqueeze(0)).square()
        )
        correlated = torch.fft.irfft(
            spectrum * gaussian_filter, n=padded_count, dim=-1
        )[:, :grid_count]
        correlated = correlated - torch.mean(correlated, dim=-1, keepdim=True)
        correlated_std = torch.std(
            correlated, dim=-1, keepdim=True, correction=0
        )
        correlated = correlated / torch.clamp(correlated_std, min=1e-16)
        values = values + std * correlated
    return values


def _detector(torch, values, grid, artifacts, generator):
    batch_size, grid_count = grid.shape
    config = artifacts.detector
    keep = torch.ones(
        (batch_size, grid_count), dtype=torch.bool, device=grid.device
    )
    if config.random_mask_probability:
        keep = keep & (
            torch.rand(
                (batch_size, grid_count),
                dtype=grid.dtype,
                device=grid.device,
                generator=generator,
            )
            >= config.random_mask_probability
        )
    for lower, upper in config.excluded_ranges:
        keep = keep & ~((grid >= lower) & (grid <= upper))
    values = torch.where(keep, values, torch.zeros_like(values))
    if config.saturation_level is not None:
        values = torch.clamp(values, max=config.saturation_level)
    if config.quantization_step is not None:
        values = (
            torch.round(values / config.quantization_step)
            * config.quantization_step
        )
    return values


def render_artifact_batch(
    positions,
    intensities,
    *,
    grid,
    artifacts: SimulationArtifacts,
    peak_mask=None,
    domain: Domain = "q",
    hkl=None,
    lattice=None,
    wavelength=None,
    profile_fwhm=None,
    profile_eta=None,
    measured_background=None,
    generator=None,
    max_entries: int = 4_194_304,
):
    """Render complete artifact-bearing patterns from a batch of powder lines.

    Inputs are padded ``[batch, peaks]`` Torch tensors plus an explicit boolean
    mask. The result is a dense ``[batch, grid]`` tensor on the same device.
    All stochastic and numerical work stays in PyTorch; peak rendering is
    chunked over reflections and never loops over batch samples.

    ``profile.model='calculator'`` has no calculator object from which to read
    a profile, so it requires ``profile_fwhm`` and optionally ``profile_eta``.
    A pre-interpolated ``measured_background`` avoids transferring and
    interpolating a :class:`BackgroundPattern` in a repeated training call.
    """

    torch = _torch()
    if not isinstance(max_entries, int) or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    peak_mask, wavelength = _validate_inputs(
        torch,
        positions,
        intensities,
        peak_mask,
        domain=domain,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        wavelength=wavelength,
    )
    grid = _grid_batch(torch, grid, positions)
    generator = _resolve_generator(torch, positions, artifacts, generator)
    centers, peak_intensities, peak_mask = _apply_peak_effects(
        torch,
        positions,
        intensities,
        peak_mask,
        domain=domain,
        artifacts=artifacts,
        hkl=hkl,
        lattice=lattice,
        generator=generator,
    )
    low_fwhm, high_fwhm, eta = _profile_parameters(
        torch,
        centers,
        domain=domain,
        artifacts=artifacts,
        wavelength=wavelength,
        profile_fwhm=profile_fwhm,
        profile_eta=profile_eta,
        generator=generator,
    )
    values = _render_split_pseudo_voigt(
        torch,
        grid,
        centers,
        peak_intensities,
        peak_mask,
        low_fwhm,
        high_fwhm,
        eta,
        max_entries=max_entries,
    )
    if artifacts.normalize_signal:
        values = _normalize(torch, values)
    values = values + _spurious_peaks(
        torch, grid, artifacts, generator, max_entries
    )
    values = values + _background(
        torch,
        grid,
        domain=domain,
        artifacts=artifacts,
        measured_background=measured_background,
        generator=generator,
    )
    values = _noise(torch, values, grid, artifacts, generator)
    values = _detector(torch, values, grid, artifacts, generator)
    if artifacts.clip_nonnegative:
        values = torch.clamp(values, min=0.0)
    if artifacts.final_normalize:
        values = _normalize(torch, values)
    return values


__all__ = ["apply_peak_artifact_batch", "render_artifact_batch"]
