"""Public Bragg diffraction calculator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from .backends.numpy_backend import NumpyBackend
from .factors import resolve_wavelength
from .hkl import HKLEnumerator
from .io import to_pmg_structure
from .profiles import GaussianProfile, GaussianProfileQ
from .results import ReflectionTable
from .renderer import apply_lp_and_multiplicity, render_profile, render_profile_q
from .structure_factor import compute_F, compute_F2, reflection_geometry
from .symmetry import SymmetryEngine


ParameterDict = Mapping[str, Any]


@dataclass
class BraggCalculator:
    """Calculate ideal kinematic X-ray or neutron powder diffraction.

    Symmetry detection and HKL enumeration happen in :meth:`load` and form a
    fixed, discrete topology.  With a Torch backend, tensors supplied through
    ``parameters=`` remain differentiable inside that topology.
    """

    mode: Literal["xray", "neutron"] = "xray"
    wavelength: float | str = 1.5406
    two_theta_range: tuple[float, float] = (10.0, 80.0)
    two_theta_step: float = 0.01
    q_range: tuple[float, float] = (0.0, 10.0)
    q_step: float = 0.005
    qmax: float | None = None
    profile: Any = field(default_factory=GaussianProfile)
    profile_q: Any = field(default_factory=GaussianProfileQ)
    backend: Any = field(default_factory=NumpyBackend)
    symprec: float = 1e-3
    angle_tolerance: float = 5.0
    primitive: bool = True
    debye_waller_factors: Mapping[str, float] = field(default_factory=dict)
    neutron_scattering_lengths: Mapping[str | int, float | str] = field(default_factory=dict)
    intensity_tolerance: float = 1e-5
    phase_chunk_entries: int = 4_194_304

    _pmg_structure: Any = field(default=None, init=False, repr=False)
    _symm: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _hkl: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"xray", "neutron"}:
            raise ValueError("mode must be 'xray' or 'neutron'")
        self.wavelength = resolve_wavelength(self.wavelength)
        self._validate_range("two_theta_range", self.two_theta_range, lower=0.0, upper=180.0)
        self._validate_range("q_range", self.q_range, lower=0.0)
        for name, value in (
            ("two_theta_step", self.two_theta_step),
            ("q_step", self.q_step),
            ("phase_chunk_entries", self.phase_chunk_entries),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.intensity_tolerance < 0 or not np.isfinite(self.intensity_tolerance):
            raise ValueError("intensity_tolerance must be non-negative and finite")
        if not np.isfinite(self.symprec) or self.symprec <= 0:
            raise ValueError("symprec must be positive and finite")
        if not np.isfinite(self.angle_tolerance) or self.angle_tolerance <= 0:
            raise ValueError("angle_tolerance must be positive and finite")
        if int(self.phase_chunk_entries) != self.phase_chunk_entries:
            raise ValueError("phase_chunk_entries must be an integer")
        self.phase_chunk_entries = int(self.phase_chunk_entries)

        physical_qmax = np.nextafter(4.0 * np.pi / self.wavelength, 0.0)
        angle_qmax = (
            4.0 * np.pi * np.sin(np.radians(self.two_theta_range[1]) / 2.0) / self.wavelength
        )
        required_qmax = min(
            max(angle_qmax, float(self.q_range[1])),
            physical_qmax,
        )
        if self.qmax is None:
            self.qmax = required_qmax
        elif not np.isfinite(self.qmax) or self.qmax <= 0:
            raise ValueError("qmax must be positive and finite")
        elif self.qmax < required_qmax - 64 * np.finfo(float).eps:
            raise ValueError(
                f"qmax={self.qmax} does not cover the configured output ranges; "
                f"use at least {required_qmax:.8g} inverse angstroms"
            )

    @staticmethod
    def _validate_range(name, values, *, lower=None, upper=None):
        if len(values) != 2:
            raise ValueError(f"{name} must contain two values")
        start, stop = map(float, values)
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError(f"{name} must be a finite increasing pair")
        if lower is not None and start < lower:
            raise ValueError(f"{name} cannot start below {lower}")
        if upper is not None and stop > upper:
            raise ValueError(f"{name} cannot end above {upper}")

    def load(self, structure_like: Any) -> "BraggCalculator":
        """Load a CIF path, pymatgen Structure, or optional ASE Atoms object."""
        self._pmg_structure = to_pmg_structure(structure_like)
        engine = SymmetryEngine(
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
            primitive=self.primitive,
        )
        self._symm = engine.reduce(self._pmg_structure)

        if self.debye_waller_factors:
            b_values = self._symm["B"].copy()
            for index, symbol in enumerate(self._symm["symbols"]):
                if symbol in self.debye_waller_factors:
                    value = float(self.debye_waller_factors[symbol])
                    if value < 0 or not np.isfinite(value):
                        raise ValueError(f"invalid Debye-Waller B value for {symbol}: {value}")
                    b_values[index] = value
            self._symm["B"] = b_values

        self._hkl = HKLEnumerator(self.wavelength, self.qmax).enumerate(
            self._symm["lattice"], self._symm["pointgroup_symbol"]
        )
        return self

    def _ensure_loaded(self):
        if not self._hkl:
            raise RuntimeError("load a structure before calculating diffraction")

    def tensor_parameters(
        self,
        requires_grad: bool | Iterable[str] = False,
    ) -> dict[str, Any]:
        """Return backend arrays suitable for the ``parameters=`` argument.

        Valid differentiable names are ``lattice``, ``frac_coords``,
        ``occupancies`` and ``b_iso``.  Species and the HKL list are discrete.
        """
        self._ensure_loaded()
        names = {"lattice", "frac_coords", "occupancies", "b_iso"}
        if isinstance(requires_grad, bool):
            grad_names = names if requires_grad else set()
        else:
            grad_names = set(requires_grad)
            unknown = grad_names - names
            if unknown:
                raise ValueError(f"unknown differentiable parameters: {sorted(unknown)}")

        values = {
            "lattice": self._symm["lattice"],
            "frac_coords": self._symm["frac_coords"],
            "occupancies": self._symm["occ"],
            "b_iso": self._symm["B"],
        }
        result = {}
        for name, value in values.items():
            array = self.backend.asarray(value, dtype=self.backend.dtype)
            if getattr(self.backend, "is_torch", False):
                array = array.clone().detach().requires_grad_(name in grad_names)
            result[name] = array
        return result

    def symmetry_coordinate_parameterization(self, *, symmetry_tolerance: float | None = None):
        """Return independent displacements that preserve prepared Wyckoff orbits."""
        from .parameters import SymmetryCoordinateParameterization

        return SymmetryCoordinateParameterization.from_calculator(
            self, symmetry_tolerance=symmetry_tolerance
        )

    def symmetry_lattice_parameterization(self, *, symmetry_tolerance: float = 1e-8):
        """Return independent log-strain modes invariant under the point group."""
        from .parameters import SymmetryLatticeParameterization

        return SymmetryLatticeParameterization.from_calculator(
            self, symmetry_tolerance=symmetry_tolerance
        )

    def _parameter_values(self, parameters: ParameterDict | None):
        parameters = {} if parameters is None else parameters
        allowed = {"lattice", "frac_coords", "occupancies", "b_iso"}
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        values = (
            parameters.get("lattice", self._symm["lattice"]),
            parameters.get("frac_coords", self._symm["frac_coords"]),
            parameters.get("occupancies", self._symm["occ"]),
            parameters.get("b_iso", self._symm["B"]),
        )
        lattice, frac, occ, b_iso = values
        atom_count = len(self._symm["Z"])
        expected = ((3, 3), (atom_count, 3), (atom_count,), (atom_count,))
        names = ("lattice", "frac_coords", "occupancies", "b_iso")
        for name, value, shape in zip(names, values, expected):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return lattice, frac, occ, b_iso

    def _domain_indices(self, domain: Literal["two_theta", "q"]):
        if domain == "two_theta":
            nominal = np.degrees(self._hkl["two_theta"])
            lower, upper = self.two_theta_range
        elif domain == "q":
            nominal = self._hkl["q"]
            lower, upper = self.q_range
        else:
            raise ValueError("domain must be 'two_theta' or 'q'")
        return np.flatnonzero((nominal >= lower) & (nominal <= upper))

    def _geometry(self, lattice, indices=None):
        hkl = self._hkl["hkl"] if indices is None else self._hkl["hkl"][indices]
        return reflection_geometry(self.backend, hkl, lattice, self.wavelength)

    def fq(self, parameters: ParameterDict | None = None, *, indices=None):
        """Return per-reciprocal-point ``|F|^2`` without powder corrections."""
        self._ensure_loaded()
        lattice, frac, occ, b_iso = self._parameter_values(parameters)
        hkl = self._hkl["hkl"] if indices is None else self._hkl["hkl"][indices]
        _, two_theta = self._geometry(lattice, indices)
        return self._compute_f2(hkl, two_theta, frac, occ, b_iso)

    def structure_factors(self, parameters: ParameterDict | None = None, *, indices=None):
        """Return complex structure factors for the fixed reciprocal topology."""
        self._ensure_loaded()
        lattice, frac, occ, b_iso = self._parameter_values(parameters)
        hkl = self._hkl["hkl"] if indices is None else self._hkl["hkl"][indices]
        _, two_theta = self._geometry(lattice, indices)
        return self._compute_f(hkl, two_theta, frac, occ, b_iso)

    def _compute_f(self, hkl, two_theta, frac, occ, b_iso):
        return compute_F(
            mode=self.mode,
            backend=self.backend,
            hkl=hkl,
            two_theta=two_theta,
            wavelength=self.wavelength,
            Z=self._symm["Z"],
            frac=frac,
            occ=occ,
            B=b_iso,
            neutron_scattering_lengths=self.neutron_scattering_lengths,
            phase_chunk_entries=self.phase_chunk_entries,
        )

    def _compute_f2(self, hkl, two_theta, frac, occ, b_iso):
        return compute_F2(
            mode=self.mode,
            backend=self.backend,
            hkl=hkl,
            two_theta=two_theta,
            wavelength=self.wavelength,
            Z=self._symm["Z"],
            frac=frac,
            occ=occ,
            B=b_iso,
            neutron_scattering_lengths=self.neutron_scattering_lengths,
            phase_chunk_entries=self.phase_chunk_entries,
        )

    def iq(
        self,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
    ):
        """Return individual reciprocal-point positions and corrected intensities."""
        indices, g, two_theta, _, _, intensity = self._individual_data(domain, parameters)
        del indices
        if domain == "two_theta":
            positions = self.backend.degrees(two_theta)
        else:
            positions = 2.0 * self.backend.pi() * g
        return positions, intensity

    def iq_components(
        self,
        wavelengths,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
    ):
        """Return several emission components while sharing one structure factor.

        For elastic diffraction, ``sin(theta) / wavelength`` and therefore the
        scattering vector are fixed by the reciprocal lattice. X-ray form
        factors and Debye--Waller terms can consequently be evaluated once;
        only the Bragg angle and angle-dependent powder correction differ
        between emission lines.
        """
        self._ensure_loaded()
        values = tuple(float(item) for item in wavelengths)
        if not values or any(not np.isfinite(item) or item <= 0 for item in values):
            raise ValueError("wavelengths must contain positive finite values")
        indices = self._domain_indices(domain)
        lattice, frac, occ, b_iso = self._parameter_values(parameters)
        g, reference_two_theta = self._geometry(lattice, indices)
        hkl = self._hkl["hkl"][indices]
        f2 = self._compute_f2(hkl, reference_two_theta, frac, occ, b_iso)
        q = 2.0 * self.backend.pi() * g
        result = []
        for wavelength in values:
            two_theta = self.backend.two_theta_from_q(q, wavelength)
            intensity = apply_lp_and_multiplicity(
                self.mode, self.backend, f2, two_theta, multiplicity=None
            )
            position = self.backend.degrees(two_theta) if domain == "two_theta" else q
            result.append((position, intensity))
        return tuple(result)

    def line_components(
        self,
        wavelengths,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
    ):
        """Return emission components with coincident reciprocal points merged.

        This path is appropriate when supplied lattice parameters preserve the
        prepared metric symmetry, as the session-level symmetry lattice
        parameterization does. Intensities remain differentiable and are
        summed without applying a reporting threshold.
        """
        patterns = self.iq_components(wavelengths, domain=domain, parameters=parameters)
        indices = self._domain_indices(domain)
        nominal = (
            np.degrees(self._hkl["two_theta"][indices])
            if domain == "two_theta"
            else self._hkl["q"][indices]
        )
        if len(nominal) == 0:
            return patterns
        tolerance = 1e-5 if domain == "two_theta" else 1e-7
        starts = np.r_[0, np.flatnonzero(np.diff(nominal) > tolerance) + 1]
        group_ids = np.cumsum(np.r_[0, (np.diff(nominal) > tolerance).astype(np.int64)])
        backend_starts = self.backend.asarray(starts, dtype=self.backend.int64)
        return tuple(
            (
                positions[backend_starts],
                self.backend.scatter_sum(intensities, group_ids, len(starts)),
            )
            for positions, intensities in patterns
        )

    def _individual_data(self, domain, parameters):
        self._ensure_loaded()
        indices = self._domain_indices(domain)
        lattice, frac, occ, b_iso = self._parameter_values(parameters)
        g, two_theta = self._geometry(lattice, indices)
        structure_factor = self._compute_f(
            self._hkl["hkl"][indices], two_theta, frac, occ, b_iso
        )
        f2 = self.backend.real(structure_factor * self.backend.conj(structure_factor))
        intensity = apply_lp_and_multiplicity(
            self.mode, self.backend, f2, two_theta, multiplicity=None
        )
        return indices, g, two_theta, structure_factor, f2, intensity

    def reflection_table(
        self,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
    ) -> ReflectionTable:
        """Return indexed per-reflection geometry and intensities."""
        indices, g, two_theta, structure_factor, f2, intensity = self._individual_data(
            domain, parameters
        )
        return ReflectionTable(
            hkl=self._hkl["hkl"][indices].copy(),
            d_spacing=1.0 / g,
            q=2.0 * self.backend.pi() * g,
            two_theta=self.backend.degrees(two_theta),
            structure_factor=structure_factor,
            f_squared=f2,
            intensity=intensity,
        )

    def line_pattern(
        self,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
        *,
        scaled: bool = False,
    ):
        """Return coincident reciprocal points merged into powder lines.

        Lattice parameters must preserve the prepared metric degeneracies when
        using this reporting method. :meth:`pattern` keeps reciprocal points
        separate and is the appropriate differentiable path for symmetry-
        breaking lattice changes.
        """
        positions, intensities = self.iq(domain=domain, parameters=parameters)
        indices = self._domain_indices(domain)
        nominal = (
            np.degrees(self._hkl["two_theta"][indices])
            if domain == "two_theta"
            else self._hkl["q"][indices]
        )
        tolerance = 1e-5 if domain == "two_theta" else 1e-7
        if len(nominal) == 0:
            return positions, intensities
        starts = np.r_[0, np.flatnonzero(np.diff(nominal) > tolerance) + 1]
        group_ids = np.cumsum(np.r_[0, (np.diff(nominal) > tolerance).astype(np.int64)])
        backend_starts = self.backend.asarray(starts, dtype=self.backend.int64)
        merged_positions = positions[backend_starts]
        merged_intensities = self.backend.scatter_sum(intensities, group_ids, len(starts))

        # Match the conventional powder-line reporting threshold.  The mask is
        # based on nominal output and is intentionally not part of the smooth
        # profile-rendering path.
        if len(starts):
            if getattr(self.backend, "is_torch", False):
                mask = (
                    merged_intensities.detach().cpu().numpy()
                    > float(merged_intensities.detach().max().cpu()) * self.intensity_tolerance
                )
            else:
                mask = merged_intensities > np.max(merged_intensities) * self.intensity_tolerance
            backend_mask = self.backend.asarray(mask, dtype=self.backend.bool)
            merged_positions = merged_positions[backend_mask]
            merged_intensities = merged_intensities[backend_mask]

        if scaled and int(merged_intensities.shape[0]):
            merged_intensities = 100.0 * merged_intensities / self.backend.max(merged_intensities)
        return merged_positions, merged_intensities

    def pattern(
        self,
        domain: Literal["two_theta", "q"] = "two_theta",
        parameters: ParameterDict | None = None,
        experiment_parameters: ParameterDict | None = None,
    ):
        """Return a gridded powder profile with optional physical nuisance controls."""
        positions, intensities = self.iq(domain=domain, parameters=parameters)
        nuisance = {} if experiment_parameters is None else dict(experiment_parameters)
        allowed = {"scale", "zero_shift", "fwhm", "background"}
        unknown = set(nuisance) - allowed
        if unknown:
            raise ValueError(f"unknown experiment parameters: {sorted(unknown)}")
        scale = nuisance.get("scale", 1.0)
        zero_shift = nuisance.get("zero_shift", 0.0)
        fwhm = nuisance.get("fwhm")
        background = nuisance.get("background", 0.0)
        positions = positions + zero_shift
        intensities = intensities * scale
        if domain == "two_theta":
            lower, upper = self.two_theta_range
            grid = self._regular_grid(lower, upper, self.two_theta_step)
            values = render_profile(
                self.profile, self.backend, grid, positions, intensities, fwhm=fwhm
            )
        elif domain == "q":
            lower, upper = self.q_range
            grid = self._regular_grid(lower, upper, self.q_step)
            values = render_profile_q(
                self.profile_q, self.backend, grid, positions, intensities, fwhm=fwhm
            )
        else:
            raise ValueError("domain must be 'two_theta' or 'q'")
        return grid, values + background

    def _regular_grid(self, lower: float, upper: float, step: float):
        intervals = int(np.floor((upper - lower) / step + 8 * np.finfo(float).eps))
        last = lower + intervals * step
        return self.backend.linspace(lower, last, intervals + 1)
