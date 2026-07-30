"""Configurable experimental effects for simulated powder patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite, log, pi, sqrt
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np


ScalarRange = float | tuple[float, float]
IntegerRange = int | tuple[int, int]
Domain = Literal["two_theta", "q"]

_FWHM_TO_SIGMA = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))


def _bounds(value: ScalarRange) -> tuple[float, float]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("ranges must contain two values")
        return float(value[0]), float(value[1])
    number = float(value)
    return number, number


def _integer_bounds(value: IntegerRange) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("integer ranges must contain two values")
        return int(value[0]), int(value[1])
    number = int(value)
    return number, number


def _validate_range(
    name: str,
    value: ScalarRange,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    try:
        lower, upper = _bounds(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number or a two-value range") from exc
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(f"{name} must be finite and increasing")
    if minimum is not None:
        invalid = lower <= minimum if strict_minimum else lower < minimum
        if invalid:
            qualifier = "greater than" if strict_minimum else "at least"
            raise ValueError(f"{name} must be {qualifier} {minimum}")
    if maximum is not None and upper > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _sample(value: ScalarRange, rng: np.random.Generator) -> float:
    lower, upper = _bounds(value)
    return lower if lower == upper else float(rng.uniform(lower, upper))


def _sample_array(
    value: ScalarRange, size: int, rng: np.random.Generator
) -> np.ndarray:
    lower, upper = _bounds(value)
    if lower == upper:
        return np.full(size, lower, dtype=float)
    return rng.uniform(lower, upper, size)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class CalibrationArtifacts:
    """Axis calibration and Bragg--Brentano specimen displacement."""

    zero_shift: ScalarRange = 0.0
    axis_scale: ScalarRange = 1.0
    peak_jitter_std: ScalarRange = 0.0
    specimen_displacement_mm: ScalarRange = 0.0
    goniometer_radius_mm: float = 200.0

    def __post_init__(self) -> None:
        _validate_range("zero_shift", self.zero_shift)
        _validate_range("axis_scale", self.axis_scale, minimum=0.0, strict_minimum=True)
        _validate_range("peak_jitter_std", self.peak_jitter_std, minimum=0.0)
        _validate_range("specimen_displacement_mm", self.specimen_displacement_mm)
        if not isfinite(self.goniometer_radius_mm) or self.goniometer_radius_mm <= 0:
            raise ValueError("goniometer_radius_mm must be positive and finite")

    def apply(self, centers, domain: Domain, backend, rng: np.random.Generator):
        scaled = centers * _sample(self.axis_scale, rng)
        shifted = scaled + _sample(self.zero_shift, rng)
        jitter_std = _sample(self.peak_jitter_std, rng)
        if jitter_std:
            shifted = shifted + backend.asarray(
                rng.normal(0.0, jitter_std, int(centers.shape[0])),
                dtype=backend.dtype,
            )
        displacement = _sample(self.specimen_displacement_mm, rng)
        if displacement:
            if domain != "two_theta":
                raise ValueError(
                    "specimen displacement is only defined for domain='two_theta'"
                )
            theta = scaled * (backend.pi() / 180.0) / 2.0
            shift_radians = (
                -2.0 * displacement * backend.cos(theta) / self.goniometer_radius_mm
            )
            shifted = shifted + backend.degrees(shift_radians)
        return shifted


@dataclass(frozen=True)
class PreferredOrientation:
    """Modified March--Dollase correction for one reciprocal-lattice axis."""

    axis: tuple[int, int, int] = (0, 0, 1)
    ratio: ScalarRange = 1.0
    fraction: ScalarRange = 1.0

    def __post_init__(self) -> None:
        if len(self.axis) != 3 or not all(isinstance(value, int) for value in self.axis):
            raise ValueError("preferred-orientation axis must contain three integers")
        if self.axis == (0, 0, 0):
            raise ValueError("preferred-orientation axis cannot be (0, 0, 0)")
        _validate_range("preferred-orientation ratio", self.ratio, minimum=0.0, strict_minimum=True)
        _validate_range(
            "preferred-orientation fraction", self.fraction, minimum=0.0, maximum=1.0
        )

    def factors(self, hkl, lattice, backend, rng: np.random.Generator):
        hkl = backend.asarray(hkl, dtype=backend.dtype)
        axis = backend.asarray(self.axis, dtype=backend.dtype)
        lattice = backend.asarray(lattice, dtype=backend.dtype)
        metric = backend.einsum("ij,kj->ik", lattice, lattice)
        reciprocal_metric = backend.inverse(metric)
        numerator = backend.einsum("hi,ij,j->h", hkl, reciprocal_metric, axis)
        hkl_norm2 = backend.einsum("hi,ij,hj->h", hkl, reciprocal_metric, hkl)
        axis_norm2 = backend.einsum("i,ij,j->", axis, reciprocal_metric, axis)
        cos2 = backend.clip(numerator**2 / (hkl_norm2 * axis_norm2), 0.0, 1.0)
        ratio = _sample(self.ratio, rng)
        fraction = _sample(self.fraction, rng)
        march = (ratio**2 * cos2 + (1.0 - cos2) / ratio) ** -1.5
        return (1.0 - fraction) + fraction * march


@dataclass(frozen=True)
class IntensityArtifacts:
    """Global, reflection-wise, and texture-dependent intensity changes."""

    scale: ScalarRange = 1.0
    peak_jitter: ScalarRange = 1.0
    peak_dropout_probability: float = 0.0
    preferred_orientation: PreferredOrientation | None = None

    def __post_init__(self) -> None:
        _validate_range("intensity scale", self.scale, minimum=0.0)
        _validate_range("peak jitter", self.peak_jitter, minimum=0.0)
        if not 0.0 <= self.peak_dropout_probability <= 1.0:
            raise ValueError("peak_dropout_probability must be between 0 and 1")
        if self.preferred_orientation is not None and not isinstance(
            self.preferred_orientation, PreferredOrientation
        ):
            raise TypeError("preferred_orientation must be a PreferredOrientation or None")

    def apply(
        self,
        intensities,
        hkl,
        lattice,
        backend,
        rng: np.random.Generator,
    ):
        values = intensities * _sample(self.scale, rng)
        peak_count = int(values.shape[0])
        if peak_count:
            jitter = _sample_array(self.peak_jitter, peak_count, rng)
            keep = rng.random(peak_count) >= self.peak_dropout_probability
            values = values * backend.asarray(jitter * keep, dtype=backend.dtype)
        if self.preferred_orientation is not None:
            values = values * self.preferred_orientation.factors(
                hkl, lattice, backend, rng
            )
        return values


@dataclass(frozen=True)
class PeakProfileArtifacts:
    """Peak broadening and asymmetry.

    ``model="calculator"`` uses the calculator's configured profile.
    ``model="pseudo_voigt"`` uses one fixed or sampled FWHM and mixing value.
    ``model="tch"`` uses angle-dependent Thompson--Cox--Hastings parameters.
    Caglioti ``u``, ``v`` and ``w`` describe squared Gaussian FWHM in degrees;
    ``x`` and ``y`` describe Lorentzian FWHM in degrees.
    """

    model: Literal["calculator", "pseudo_voigt", "tch"] = "calculator"
    fwhm: ScalarRange = 0.1
    eta: ScalarRange = 0.5
    caglioti_u: ScalarRange = 0.0
    caglioti_v: ScalarRange = 0.0
    caglioti_w: ScalarRange = 0.01
    lorentzian_x: ScalarRange = 0.0
    lorentzian_y: ScalarRange = 0.0
    crystallite_size_nm: ScalarRange | None = None
    scherrer_constant: float = 0.9
    microstrain: ScalarRange = 0.0
    axial_asymmetry: ScalarRange = 0.0

    def __post_init__(self) -> None:
        if self.model not in {"calculator", "pseudo_voigt", "tch"}:
            raise ValueError("profile model must be 'calculator', 'pseudo_voigt', or 'tch'")
        _validate_range("fwhm", self.fwhm, minimum=0.0, strict_minimum=True)
        _validate_range("eta", self.eta, minimum=0.0, maximum=1.0)
        _validate_range("caglioti_u", self.caglioti_u)
        _validate_range("caglioti_v", self.caglioti_v)
        _validate_range("caglioti_w", self.caglioti_w)
        _validate_range("lorentzian_x", self.lorentzian_x, minimum=0.0)
        _validate_range("lorentzian_y", self.lorentzian_y, minimum=0.0)
        _validate_range("microstrain", self.microstrain, minimum=0.0)
        _validate_range("axial_asymmetry", self.axial_asymmetry, minimum=0.0)
        if self.crystallite_size_nm is not None:
            _validate_range(
                "crystallite_size_nm",
                self.crystallite_size_nm,
                minimum=0.0,
                strict_minimum=True,
            )
        if not isfinite(self.scherrer_constant) or self.scherrer_constant <= 0:
            raise ValueError("scherrer_constant must be positive and finite")
        if self.model != "tch" and (
            self.crystallite_size_nm is not None or _bounds(self.microstrain)[1] > 0
        ):
            raise ValueError(
                "crystallite-size and microstrain broadening require model='tch'"
            )
        if self.model == "calculator" and _bounds(self.axial_asymmetry)[1] > 0:
            raise ValueError("axial_asymmetry requires an explicit profile model")

    def render(
        self,
        calculator,
        domain: Domain,
        grid,
        centers,
        intensities,
        backend,
        rng: np.random.Generator,
    ):
        if self.model == "calculator":
            return _render_default(calculator, domain, grid, centers, intensities)

        if self.model == "pseudo_voigt":
            fwhm = centers * 0.0 + _sample(self.fwhm, rng)
            eta = centers * 0.0 + _sample(self.eta, rng)
        else:
            fwhm, eta = self._tch_widths(
                calculator, domain, centers, backend, rng
            )

        asymmetry = _sample(self.axial_asymmetry, rng)
        if asymmetry:
            if domain != "two_theta":
                raise ValueError("axial_asymmetry is only defined for domain='two_theta'")
            theta = centers * (backend.pi() / 180.0) / 2.0
            tangent = backend.clip(
                backend.sin(theta) / backend.cos(theta), 1e-8, float("inf")
            )
            low_fwhm = fwhm * (1.0 + asymmetry / tangent)
        else:
            low_fwhm = fwhm
        return _render_split_pseudo_voigt(
            grid,
            centers,
            intensities,
            low_fwhm,
            fwhm,
            eta,
            backend,
            _max_entries(calculator, domain),
        )

    def _tch_widths(
        self,
        calculator,
        domain: Domain,
        centers,
        backend,
        rng: np.random.Generator,
    ):
        if domain == "two_theta":
            two_theta = centers * (backend.pi() / 180.0)
        else:
            two_theta = backend.two_theta_from_q(centers, calculator.wavelength)
        theta = two_theta / 2.0
        tangent = backend.sin(theta) / backend.cos(theta)

        u = _sample(self.caglioti_u, rng)
        v = _sample(self.caglioti_v, rng)
        w = _sample(self.caglioti_w, rng)
        gaussian2 = u * tangent**2 + v * tangent + w
        if np.any(_to_numpy(gaussian2) <= 0):
            raise ValueError("Caglioti U, V, W produce a non-positive Gaussian width")
        gaussian = backend.sqrt(gaussian2)

        x = _sample(self.lorentzian_x, rng)
        y = _sample(self.lorentzian_y, rng)
        lorentzian = x / backend.cos(theta) + y * tangent

        size_nm = (
            None
            if self.crystallite_size_nm is None
            else _sample(self.crystallite_size_nm, rng)
        )
        if size_nm is not None:
            size_angstrom = 10.0 * size_nm
            size_radians = (
                self.scherrer_constant
                * calculator.wavelength
                / (size_angstrom * backend.cos(theta))
            )
            lorentzian = lorentzian + backend.degrees(size_radians)

        microstrain = _sample(self.microstrain, rng)
        if microstrain:
            lorentzian = lorentzian + backend.degrees(4.0 * microstrain * tangent)
        lorentzian = backend.clip(lorentzian, 0.0, float("inf"))

        combined_fifth = (
            gaussian**5
            + 2.69269 * gaussian**4 * lorentzian
            + 2.42843 * gaussian**3 * lorentzian**2
            + 4.47163 * gaussian**2 * lorentzian**3
            + 0.07842 * gaussian * lorentzian**4
            + lorentzian**5
        )
        fwhm_degrees = combined_fifth**0.2
        ratio = lorentzian / fwhm_degrees
        eta = backend.clip(
            1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3,
            0.0,
            1.0,
        )
        if domain == "q":
            dq_ddegree = (
                2.0
                * backend.pi()
                * backend.cos(theta)
                / calculator.wavelength
                * (backend.pi() / 180.0)
            )
            return fwhm_degrees * dq_ddegree, eta
        return fwhm_degrees, eta


@dataclass(frozen=True)
class AmorphousHump:
    """Broad Gaussian or pseudo-Voigt contribution specified by peak height."""

    center: ScalarRange
    fwhm: ScalarRange
    height: ScalarRange
    eta: ScalarRange = 0.0

    def __post_init__(self) -> None:
        _validate_range("amorphous-hump center", self.center)
        _validate_range("amorphous-hump fwhm", self.fwhm, minimum=0.0, strict_minimum=True)
        _validate_range("amorphous-hump height", self.height, minimum=0.0)
        _validate_range("amorphous-hump eta", self.eta, minimum=0.0, maximum=1.0)


@dataclass(frozen=True)
class BackgroundPattern:
    """Measured background samples with source and checksum provenance."""

    coordinate: np.ndarray
    intensity: np.ndarray
    domain: Domain
    uncertainty: np.ndarray | None = None
    source: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        coordinate = np.array(self.coordinate, dtype=np.float64, copy=True)
        intensity = np.array(self.intensity, dtype=np.float64, copy=True)
        uncertainty = (
            None
            if self.uncertainty is None
            else np.array(self.uncertainty, dtype=np.float64, copy=True)
        )
        if self.domain not in {"two_theta", "q"}:
            raise ValueError("background domain must be 'two_theta' or 'q'")
        if coordinate.ndim != 1 or intensity.shape != coordinate.shape or len(coordinate) < 2:
            raise ValueError("background coordinate and intensity must be equal 1D arrays")
        if not np.all(np.isfinite(coordinate)) or not np.all(np.diff(coordinate) > 0):
            raise ValueError("background coordinates must be finite and strictly increasing")
        if not np.all(np.isfinite(intensity)) or np.any(intensity < 0):
            raise ValueError("background intensities must be finite and non-negative")
        if uncertainty is not None:
            if uncertainty.shape != coordinate.shape:
                raise ValueError("background uncertainty must match the coordinate shape")
            if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
                raise ValueError("background uncertainty must be positive and finite")
            uncertainty.setflags(write=False)
        coordinate.setflags(write=False)
        intensity.setflags(write=False)
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "uncertainty", uncertainty)

    @classmethod
    def from_file(
        cls,
        path: str | PathLike,
        *,
        domain: Domain,
        third_column: Literal["sigma", "weight", "ignore"] = "sigma",
        source: str | None = None,
    ) -> "BackgroundPattern":
        """Load whitespace- or comma-separated ``.xy``/``.xye`` samples."""
        file_path = Path(path)
        if file_path.suffix.lower() not in {".xy", ".xye", ".txt", ".dat"}:
            raise ValueError("background files must use .xy, .xye, .txt, or .dat")
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        raw = file_path.read_bytes()
        rows: list[list[float]] = []
        for line_number, raw_line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "!", ";")):
                continue
            fields = line.replace(",", " ").split()
            if len(fields) < 2:
                raise ValueError(f"{file_path}:{line_number}: expected at least two columns")
            try:
                rows.append([float(value) for value in fields[:3]])
            except ValueError as exc:
                raise ValueError(
                    f"{file_path}:{line_number}: background columns must be numeric"
                ) from exc
        if len(rows) < 2:
            raise ValueError("background file must contain at least two data rows")
        if third_column not in {"sigma", "weight", "ignore"}:
            raise ValueError("third_column must be 'sigma', 'weight', or 'ignore'")
        if third_column != "ignore" and any(len(row) < 3 for row in rows):
            if file_path.suffix.lower() == ".xye":
                raise ValueError(".xye background files require a third column")
            uncertainty = None
        elif third_column == "sigma":
            uncertainty = np.array([row[2] for row in rows], dtype=float)
        elif third_column == "weight":
            weights = np.array([row[2] for row in rows], dtype=float)
            if np.any(weights <= 0):
                raise ValueError("background weights must be positive")
            uncertainty = 1.0 / np.sqrt(weights)
        else:
            uncertainty = None
        return cls(
            coordinate=np.array([row[0] for row in rows]),
            intensity=np.array([row[1] for row in rows]),
            uncertainty=uncertainty,
            domain=domain,
            source=source or str(file_path),
            source_sha256=sha256(raw).hexdigest(),
        )

    def interpolate(
        self,
        grid,
        domain: Domain,
        *,
        extrapolation: Literal["error", "zero", "edge"] = "error",
    ) -> np.ndarray:
        if domain != self.domain:
            raise ValueError(
                f"background uses domain={self.domain!r}, but the pattern uses {domain!r}"
            )
        target = np.asarray(_to_numpy(grid), dtype=float)
        if extrapolation == "error":
            tolerance = 16 * np.finfo(float).eps * max(1.0, np.max(np.abs(target)))
            if (
                target[0] < self.coordinate[0] - tolerance
                or target[-1] > self.coordinate[-1] + tolerance
            ):
                raise ValueError("background samples do not cover the simulation grid")
            left = float(self.intensity[0])
            right = float(self.intensity[-1])
        elif extrapolation == "zero":
            left = right = 0.0
        elif extrapolation == "edge":
            left = float(self.intensity[0])
            right = float(self.intensity[-1])
        else:
            raise ValueError("extrapolation must be 'error', 'zero', or 'edge'")
        return np.interp(target, self.coordinate, self.intensity, left=left, right=right)


@dataclass(frozen=True)
class BackgroundLibrary:
    """Checksum-verified collection of sourced measured backgrounds."""

    manifest_path: Path

    def __post_init__(self) -> None:
        path = Path(self.manifest_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        object.__setattr__(self, "manifest_path", path)
        self._records()

    @classmethod
    def bundled(cls) -> "BackgroundLibrary":
        return cls(Path(__file__).with_name("background_library") / "manifest.json")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._records()))

    def load(self, name: str) -> BackgroundPattern:
        records = self._records()
        if name not in records:
            available = ", ".join(sorted(records)) or "none"
            raise KeyError(f"unknown background {name!r}; available backgrounds: {available}")
        record = records[name]
        required = {"path", "domain", "source", "sha256"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"background {name!r} is missing manifest fields: {sorted(missing)}")
        if not isinstance(record["source"], str) or not record["source"].strip():
            raise ValueError(f"background {name!r} must declare a non-empty source")
        root = self.manifest_path.parent.resolve()
        data_path = (root / record["path"]).resolve()
        if root not in data_path.parents:
            raise ValueError(f"background {name!r} escapes the library directory")
        pattern = BackgroundPattern.from_file(
            data_path,
            domain=record["domain"],
            third_column=record.get("third_column", "sigma"),
            source=record["source"],
        )
        if pattern.source_sha256 != record["sha256"]:
            raise ValueError(f"background {name!r} does not match its SHA-256 digest")
        return pattern

    def _records(self) -> dict:
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid background manifest: {self.manifest_path}") from exc
        if manifest.get("schema_version") != 1 or not isinstance(
            manifest.get("backgrounds"), dict
        ):
            raise ValueError("background manifest must contain schema_version=1 and backgrounds")
        if not all(
            isinstance(name, str) and isinstance(record, dict)
            for name, record in manifest["backgrounds"].items()
        ):
            raise ValueError("background manifest entries must be named JSON objects")
        return manifest["backgrounds"]


@dataclass(frozen=True)
class BackgroundArtifacts:
    """Analytical, amorphous, and measured background contributions."""

    constant: ScalarRange = 0.0
    linear_slope: ScalarRange = 0.0
    chebyshev_coefficients: tuple[float, ...] = ()
    amorphous_humps: tuple[AmorphousHump, ...] = ()
    measured: BackgroundPattern | None = None
    measured_scale: ScalarRange = 1.0
    measured_offset: ScalarRange = 0.0
    extrapolation: Literal["error", "zero", "edge"] = "error"

    def __post_init__(self) -> None:
        _validate_range("background constant", self.constant, minimum=0.0)
        _validate_range("background linear_slope", self.linear_slope)
        _validate_range("measured background scale", self.measured_scale, minimum=0.0)
        _validate_range("measured background offset", self.measured_offset)
        if not all(isfinite(value) for value in self.chebyshev_coefficients):
            raise ValueError("Chebyshev background coefficients must be finite")
        if not all(isinstance(hump, AmorphousHump) for hump in self.amorphous_humps):
            raise TypeError("amorphous_humps must contain AmorphousHump objects")
        if self.measured is not None and not isinstance(self.measured, BackgroundPattern):
            raise TypeError("measured background must be a BackgroundPattern or None")
        if self.extrapolation not in {"error", "zero", "edge"}:
            raise ValueError("extrapolation must be 'error', 'zero', or 'edge'")

    def values(
        self,
        grid,
        domain: Domain,
        backend,
        rng: np.random.Generator,
    ):
        result = backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
        center = 0.5 * (grid[0] + grid[-1])
        result = result + _sample(self.constant, rng)
        result = result + _sample(self.linear_slope, rng) * (grid - center)

        if self.chebyshev_coefficients:
            span = float(_to_numpy(grid[-1] - grid[0]))
            if span:
                normalized = 2.0 * (grid - grid[0]) / (grid[-1] - grid[0]) - 1.0
            else:
                normalized = grid * 0.0
            t0 = backend.ones((int(grid.shape[0]),), dtype=backend.dtype)
            result = result + self.chebyshev_coefficients[0] * t0
            if len(self.chebyshev_coefficients) > 1:
                t1 = normalized
                result = result + self.chebyshev_coefficients[1] * t1
                for coefficient in self.chebyshev_coefficients[2:]:
                    t2 = 2.0 * normalized * t1 - t0
                    result = result + coefficient * t2
                    t0, t1 = t1, t2

        for hump in self.amorphous_humps:
            center_value = _sample(hump.center, rng)
            width = _sample(hump.fwhm, rng)
            height = _sample(hump.height, rng)
            eta = _sample(hump.eta, rng)
            delta = grid - center_value
            gaussian = backend.exp(-0.5 * (delta / (width * _FWHM_TO_SIGMA)) ** 2)
            lorentzian = 1.0 / (1.0 + (2.0 * delta / width) ** 2)
            result = result + height * ((1.0 - eta) * gaussian + eta * lorentzian)

        if self.measured is not None:
            measured = self.measured.interpolate(
                grid, domain, extrapolation=self.extrapolation
            )
            measured = (
                _sample(self.measured_scale, rng) * measured
                + _sample(self.measured_offset, rng)
            )
            result = result + backend.asarray(measured, dtype=backend.dtype)
        return result


@dataclass(frozen=True)
class SpuriousPeakArtifacts:
    """Unindexed peaks, expressed as integrated pseudo-Voigt areas."""

    count: IntegerRange = 0
    intensity: ScalarRange = (0.01, 0.1)
    fwhm: ScalarRange = 0.05
    eta: ScalarRange = 0.5

    def __post_init__(self) -> None:
        try:
            count_min, count_max = _integer_bounds(self.count)
        except (TypeError, ValueError) as exc:
            raise ValueError("spurious peak count must be an integer or integer range") from exc
        if count_min < 0 or count_min > count_max:
            raise ValueError("spurious peak count must be non-negative and increasing")
        _validate_range("spurious peak intensity", self.intensity, minimum=0.0)
        _validate_range("spurious peak fwhm", self.fwhm, minimum=0.0, strict_minimum=True)
        _validate_range("spurious peak eta", self.eta, minimum=0.0, maximum=1.0)

    def render(self, grid, backend, rng: np.random.Generator, max_entries: int):
        count_min, count_max = _integer_bounds(self.count)
        count = int(rng.integers(count_min, count_max + 1))
        if not count:
            return backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
        grid_numpy = _to_numpy(grid)
        centers = backend.asarray(
            rng.uniform(float(grid_numpy[0]), float(grid_numpy[-1]), count),
            dtype=backend.dtype,
        )
        intensities = backend.asarray(
            _sample_array(self.intensity, count, rng), dtype=backend.dtype
        )
        widths = backend.asarray(
            _sample_array(self.fwhm, count, rng), dtype=backend.dtype
        )
        eta = backend.asarray(_sample_array(self.eta, count, rng), dtype=backend.dtype)
        return _render_split_pseudo_voigt(
            grid,
            centers,
            intensities,
            widths,
            widths,
            eta,
            backend,
            max_entries,
        )


@dataclass(frozen=True)
class NoiseArtifacts:
    """Independent, correlated, and Poisson counting noise."""

    gaussian_std: ScalarRange = 0.0
    correlated_std: ScalarRange = 0.0
    correlation_length: ScalarRange = 0.1
    poisson_count_scale: ScalarRange | None = None

    def __post_init__(self) -> None:
        _validate_range("gaussian_std", self.gaussian_std, minimum=0.0)
        _validate_range("correlated_std", self.correlated_std, minimum=0.0)
        _validate_range(
            "correlation_length", self.correlation_length, minimum=0.0, strict_minimum=True
        )
        if self.poisson_count_scale is not None:
            _validate_range(
                "poisson_count_scale",
                self.poisson_count_scale,
                minimum=0.0,
                strict_minimum=True,
            )

    def apply(self, values, grid, backend, rng: np.random.Generator):
        if self.poisson_count_scale is not None:
            count_scale = _sample(self.poisson_count_scale, rng)
            expected = np.clip(_to_numpy(values), 0.0, None) * count_scale
            values = backend.asarray(
                rng.poisson(expected).astype(float) / count_scale,
                dtype=backend.dtype,
            )

        gaussian_std = _sample(self.gaussian_std, rng)
        if gaussian_std:
            noise = rng.normal(0.0, gaussian_std, int(grid.shape[0]))
            values = values + backend.asarray(noise, dtype=backend.dtype)

        correlated_std = _sample(self.correlated_std, rng)
        if correlated_std:
            coordinate = _to_numpy(grid)
            if len(coordinate) < 2:
                raise ValueError("correlated noise requires at least two grid points")
            step = float(np.median(np.diff(coordinate)))
            sigma_points = _sample(self.correlation_length, rng) / step
            radius = min(
                max(1, int(np.ceil(4.0 * sigma_points))),
                max(1, (len(coordinate) - 1) // 2),
            )
            offsets = np.arange(-radius, radius + 1, dtype=float)
            kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
            kernel /= np.sqrt(np.sum(kernel**2))
            correlated = np.convolve(rng.normal(size=len(coordinate)), kernel, mode="same")
            correlated *= correlated_std / max(float(np.std(correlated)), 1e-16)
            values = values + backend.asarray(correlated, dtype=backend.dtype)
        return values


@dataclass(frozen=True)
class DetectorArtifacts:
    """Missing channels, saturation, and intensity quantization."""

    random_mask_probability: float = 0.0
    excluded_ranges: tuple[tuple[float, float], ...] = ()
    saturation_level: float | None = None
    quantization_step: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.random_mask_probability <= 1.0:
            raise ValueError("random_mask_probability must be between 0 and 1")
        for lower, upper in self.excluded_ranges:
            if not isfinite(lower) or not isfinite(upper) or lower >= upper:
                raise ValueError("excluded detector ranges must be finite and increasing")
        if self.saturation_level is not None and (
            not isfinite(self.saturation_level) or self.saturation_level <= 0
        ):
            raise ValueError("saturation_level must be positive and finite")
        if self.quantization_step is not None and (
            not isfinite(self.quantization_step) or self.quantization_step <= 0
        ):
            raise ValueError("quantization_step must be positive and finite")

    def apply(self, values, grid, backend, rng: np.random.Generator):
        keep = np.ones(int(grid.shape[0]), dtype=bool)
        coordinate = _to_numpy(grid)
        if self.random_mask_probability:
            keep &= rng.random(len(keep)) >= self.random_mask_probability
        for lower, upper in self.excluded_ranges:
            keep &= ~((coordinate >= lower) & (coordinate <= upper))
        values = values * backend.asarray(keep, dtype=backend.dtype)
        if self.saturation_level is not None:
            values = backend.clip(values, -float("inf"), self.saturation_level)
        if self.quantization_step is not None:
            values = backend.round(values / self.quantization_step) * self.quantization_step
        return values


@dataclass(frozen=True)
class SimulationArtifacts:
    """Complete, independently configurable experimental-effect model."""

    calibration: CalibrationArtifacts = field(default_factory=CalibrationArtifacts)
    profile: PeakProfileArtifacts = field(default_factory=PeakProfileArtifacts)
    intensity: IntensityArtifacts = field(default_factory=IntensityArtifacts)
    background: BackgroundArtifacts = field(default_factory=BackgroundArtifacts)
    noise: NoiseArtifacts = field(default_factory=NoiseArtifacts)
    detector: DetectorArtifacts = field(default_factory=DetectorArtifacts)
    spurious_peaks: SpuriousPeakArtifacts = field(default_factory=SpuriousPeakArtifacts)
    normalize_signal: bool = False
    clip_nonnegative: bool = True
    final_normalize: bool = False
    domain: Domain | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        components = (
            ("calibration", self.calibration, CalibrationArtifacts),
            ("profile", self.profile, PeakProfileArtifacts),
            ("intensity", self.intensity, IntensityArtifacts),
            ("background", self.background, BackgroundArtifacts),
            ("noise", self.noise, NoiseArtifacts),
            ("detector", self.detector, DetectorArtifacts),
            ("spurious_peaks", self.spurious_peaks, SpuriousPeakArtifacts),
        )
        for name, value, expected in components:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be a {expected.__name__}")
        if self.domain not in {None, "two_theta", "q"}:
            raise ValueError("domain must be None, 'two_theta', or 'q'")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer or None")

    def apply(
        self,
        calculator,
        domain: Domain,
        grid,
        centers,
        intensities,
        *,
        hkl,
        lattice,
    ):
        """Apply this configuration to one simulated pattern."""
        if self.domain is not None and domain != self.domain:
            raise ValueError(
                f"this artifact configuration requires domain={self.domain!r}, got {domain!r}"
            )
        backend = calculator.backend
        rng = np.random.default_rng(self.seed)
        centers = self.calibration.apply(centers, domain, backend, rng)
        intensities = self.intensity.apply(
            intensities, hkl, lattice, backend, rng
        )
        values = self.profile.render(
            calculator, domain, grid, centers, intensities, backend, rng
        )
        if self.normalize_signal:
            values = _normalize(values, backend)
        values = values + self.spurious_peaks.render(
            grid, backend, rng, _max_entries(calculator, domain)
        )
        values = values + self.background.values(grid, domain, backend, rng)
        values = self.noise.apply(values, grid, backend, rng)
        values = self.detector.apply(values, grid, backend, rng)
        if self.clip_nonnegative:
            values = backend.clip(values, 0.0, float("inf"))
        if self.final_normalize:
            values = _normalize(values, backend)
        return values


def _max_entries(calculator, domain: Domain) -> int:
    profile = calculator.profile if domain == "two_theta" else calculator.profile_q
    return int(getattr(profile, "max_entries", 4_194_304))


def _render_default(calculator, domain, grid, centers, intensities):
    if domain == "two_theta":
        return calculator.profile.render(grid, centers, intensities, calculator.backend)
    return calculator.profile_q.render(grid, centers, intensities, calculator.backend)


def _render_split_pseudo_voigt(
    grid,
    centers,
    intensities,
    low_fwhm,
    high_fwhm,
    eta,
    backend,
    max_entries,
):
    """Render area-normalized split pseudo-Voigt peaks."""
    if int(centers.shape[0]) == 0:
        return backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    peak_chunk = max(1, max_entries // max(int(grid.shape[0]), 1))
    result = backend.zeros((int(grid.shape[0]),), dtype=backend.dtype)
    gaussian_integral = sqrt(pi) / (4.0 * sqrt(log(2.0)))
    for start in range(0, int(centers.shape[0]), peak_chunk):
        stop = min(start + peak_chunk, int(centers.shape[0]))
        offset = grid[:, None] - centers[None, start:stop]
        low = low_fwhm[start:stop][None, :]
        high = high_fwhm[start:stop][None, :]
        width = backend.where(offset < 0.0, low, high)
        gaussian = backend.exp(-4.0 * log(2.0) * (offset / width) ** 2)
        gaussian = gaussian / (gaussian_integral * (low + high))
        lorentzian = 1.0 / (1.0 + 4.0 * (offset / width) ** 2)
        lorentzian = lorentzian / (0.25 * pi * (low + high))
        mixing = eta[start:stop][None, :]
        shape = (1.0 - mixing) * gaussian + mixing * lorentzian
        result = result + backend.sum(
            intensities[None, start:stop] * shape, axis=1
        )
    return result


def _normalize(values, backend):
    return values / (backend.max(values) + 1e-16)
