"""Symmetry-compatible continuous parameterizations for refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _periodic_difference(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return difference - np.rint(difference)


def _null_space(matrix, tolerance: float) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1])
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    scale = singular_values[0] if len(singular_values) else 0.0
    rank = int(np.count_nonzero(singular_values > tolerance * max(scale, 1.0)))
    basis = right_vectors[rank:].T.copy()
    for column in range(basis.shape[1]):
        nonzero = np.flatnonzero(np.abs(basis[:, column]) > tolerance)
        if len(nonzero) and basis[nonzero[0], column] < 0:
            basis[:, column] *= -1
    return basis


def _symmetric_matrix_basis() -> np.ndarray:
    """Return a Frobenius-orthonormal basis for symmetric 3 by 3 matrices."""
    basis = []
    for row, column in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[row, column] = 1.0
        matrix[column, row] = 1.0
        if row != column:
            matrix /= np.sqrt(2.0)
        basis.append(matrix)
    return np.asarray(basis)


def _symmetric_coefficients(matrix: np.ndarray) -> np.ndarray:
    basis = _symmetric_matrix_basis()
    return np.einsum("kij,ij->k", basis, matrix)


def lattice_parameters(lattice) -> dict[str, float]:
    """Return conventional lengths and angles from a row-vector lattice."""
    matrix = np.asarray(lattice, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("lattice must have shape (3, 3)")
    lengths = np.linalg.norm(matrix, axis=1)

    def angle(left, right):
        cosine = np.dot(matrix[left], matrix[right]) / (lengths[left] * lengths[right])
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    return {
        "a": float(lengths[0]),
        "b": float(lengths[1]),
        "c": float(lengths[2]),
        "alpha": angle(1, 2),
        "beta": angle(0, 2),
        "gamma": angle(0, 1),
    }


@dataclass(frozen=True)
class SymmetryLatticeParameterization:
    """Point-group-invariant positive-volume lattice deformation.

    Independent values are dimensionless log-strain coordinates. The strain
    tensor is constrained to the invariant symmetric subspace of the loaded
    point group, and its matrix exponential is applied in Cartesian space.
    This gives the expected 1, 2, 3, 4 and 6 metric degrees of freedom for
    cubic, uniaxial, orthorhombic, monoclinic and triclinic structures.
    """

    base_lattice: np.ndarray
    strain_basis: np.ndarray
    labels: tuple[str, ...]
    crystal_system: str
    strain_scale: float
    symmetry_tolerance: float

    @classmethod
    def from_calculator(
        cls,
        calculator,
        *,
        symmetry_tolerance: float = 1e-8,
        strain_scale: float = 0.01,
    ) -> "SymmetryLatticeParameterization":
        calculator._ensure_loaded()
        tolerance = float(symmetry_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("symmetry_tolerance must be positive and finite")
        if not np.isfinite(strain_scale) or strain_scale <= 0:
            raise ValueError("strain_scale must be positive and finite")

        metadata = calculator._symm
        lattice = np.asarray(metadata["lattice"], dtype=np.float64)
        inverse_cartesian_lattice = np.linalg.inv(lattice.T)
        matrix_basis = _symmetric_matrix_basis()
        constraints = []
        for fractional_rotation in np.asarray(metadata["symm_rot"], dtype=np.float64):
            cartesian_rotation = lattice.T @ fractional_rotation @ inverse_cartesian_lattice
            transformed = np.asarray(
                [cartesian_rotation @ item @ cartesian_rotation.T for item in matrix_basis]
            )
            transform = np.stack([_symmetric_coefficients(item) for item in transformed], axis=1)
            constraints.append(transform - np.eye(6))
        constraint_matrix = np.concatenate(constraints, axis=0)
        coefficients = _null_space(constraint_matrix, tolerance)
        strain_basis = np.einsum("ki,kjl->ijl", coefficients, matrix_basis)
        crystal_system = str(metadata["crystal_system"])
        labels = tuple(
            f"{crystal_system}.metric_mode_{index + 1}" for index in range(strain_basis.shape[0])
        )
        return cls(
            base_lattice=lattice.copy(),
            strain_basis=strain_basis,
            labels=labels,
            crystal_system=crystal_system,
            strain_scale=float(strain_scale),
            symmetry_tolerance=tolerance,
        )

    @property
    def independent_count(self) -> int:
        return int(self.strain_basis.shape[0])

    def initial_values(self, backend, *, requires_grad: bool = False):
        values = backend.zeros((self.independent_count,), dtype=backend.dtype)
        if getattr(backend, "is_torch", False):
            values = values.clone().detach().requires_grad_(requires_grad)
        elif requires_grad:
            raise TypeError("requires_grad is available only with TorchBackend")
        return values

    def expand(self, independent_values, backend):
        if tuple(independent_values.shape) != (self.independent_count,):
            raise ValueError(
                f"independent_values must have shape ({self.independent_count},), "
                f"got {tuple(independent_values.shape)}"
            )
        basis = backend.asarray(self.strain_basis, dtype=backend.dtype)
        log_strain = self.strain_scale * backend.einsum("k,kij->ij", independent_values, basis)
        deformation = backend.matrix_exp(log_strain)
        base = backend.asarray(self.base_lattice, dtype=backend.dtype)
        return backend.matmul(base, deformation)

    def physical_parameters(self, independent_values) -> dict[str, float]:
        from .backends import NumpyBackend

        values = np.asarray(independent_values, dtype=np.float64)
        lattice = self.expand(values, NumpyBackend())
        return lattice_parameters(lattice)


@dataclass(frozen=True)
class OccupancyOrbitSpec:
    """One symmetry orbit represented by a composition or vacancy simplex."""

    orbit_index: int
    representative_site: int
    member_sites: np.ndarray
    atomic_numbers: tuple[int, ...]
    symbols: tuple[str, ...]
    total_occupancy: float
    includes_vacancy: bool
    reference_component: int
    parameter_slice: slice

    @property
    def species_count(self) -> int:
        return len(self.atomic_numbers)

    @property
    def component_count(self) -> int:
        return self.species_count + int(self.includes_vacancy)

    @property
    def degrees_of_freedom(self) -> int:
        return self.component_count - 1


@dataclass(frozen=True)
class SymmetryOccupancyParameterization:
    """Shared-site occupancy simplexes propagated over crystallographic orbits."""

    mode: str
    base_occupancies: np.ndarray
    contribution_orbits: np.ndarray
    contribution_components: np.ndarray
    orbits: tuple[OccupancyOrbitSpec, ...]
    labels: tuple[str, ...]
    initial_raw_values: np.ndarray
    probability_floor: float

    @classmethod
    def from_calculator(
        cls,
        calculator,
        *,
        mode: str = "composition",
        probability_floor: float = 1e-6,
    ) -> "SymmetryOccupancyParameterization":
        calculator._ensure_loaded()
        if mode not in {"composition", "vacancy"}:
            raise ValueError("occupancy mode must be 'composition' or 'vacancy'")
        if not np.isfinite(probability_floor) or not 0 < probability_floor < 0.1:
            raise ValueError("probability_floor must lie between zero and 0.1")

        metadata = calculator._symm
        site_indices = np.asarray(metadata["site_indices"], dtype=np.int64)
        atomic_numbers = np.asarray(metadata["Z"], dtype=np.int64)
        symbols = np.asarray(metadata["symbols"], dtype=object)
        occupancies = np.asarray(metadata["occ"], dtype=np.float64)
        orbit_members = tuple(
            np.asarray(item, dtype=np.int64) for item in metadata["orbit_indices"]
        )
        contribution_orbits = np.full(len(occupancies), -1, dtype=np.int64)
        contribution_components = np.full(len(occupancies), -1, dtype=np.int64)
        specifications = []
        labels = []
        raw_values = []
        parameter_count = 0

        for orbit_index, members in enumerate(orbit_members):
            representative = int(members[0])
            representative_contributions = np.flatnonzero(site_indices == representative)
            orbit_z = tuple(int(item) for item in atomic_numbers[representative_contributions])
            orbit_symbols = tuple(str(item) for item in symbols[representative_contributions])
            orbit_occ = occupancies[representative_contributions]
            total = float(np.sum(orbit_occ))
            if total <= 0 or total > 1.0 + 1e-8:
                raise ValueError(f"orbit {orbit_index} has invalid total occupancy {total}")

            for member in members:
                contributions = np.flatnonzero(site_indices == member)
                member_z = tuple(int(item) for item in atomic_numbers[contributions])
                if member_z != orbit_z:
                    raise ValueError(
                        f"orbit {orbit_index} has inconsistent shared-site species ordering"
                    )
                if not np.allclose(occupancies[contributions], orbit_occ, rtol=0, atol=1e-8):
                    raise ValueError(f"orbit {orbit_index} has inconsistent member occupancies")
                contribution_orbits[contributions] = orbit_index
                contribution_components[contributions] = np.arange(len(contributions))

            includes_vacancy = mode == "vacancy"
            probabilities = (
                np.r_[orbit_occ, max(0.0, 1.0 - total)] if includes_vacancy else orbit_occ / total
            )
            probabilities = np.maximum(probabilities, probability_floor)
            probabilities /= probabilities.sum()
            reference = int(np.argmax(probabilities))
            non_reference = [index for index in range(len(probabilities)) if index != reference]
            start = parameter_count
            parameter_count += len(non_reference)
            parameter_slice = slice(start, parameter_count)
            component_names = list(orbit_symbols) + (["vacancy"] if includes_vacancy else [])
            raw_values.extend(
                float(np.log(probabilities[index] / probabilities[reference]))
                for index in non_reference
            )
            labels.extend(
                f"orbit_{orbit_index}.{component_names[index]}_vs_{component_names[reference]}"
                for index in non_reference
            )
            specifications.append(
                OccupancyOrbitSpec(
                    orbit_index=orbit_index,
                    representative_site=representative,
                    member_sites=members.copy(),
                    atomic_numbers=orbit_z,
                    symbols=orbit_symbols,
                    total_occupancy=total,
                    includes_vacancy=includes_vacancy,
                    reference_component=reference,
                    parameter_slice=parameter_slice,
                )
            )

        if np.any(contribution_orbits < 0) or np.any(contribution_components < 0):
            raise RuntimeError("not every scattering contribution was assigned to an orbit")
        return cls(
            mode=mode,
            base_occupancies=occupancies.copy(),
            contribution_orbits=contribution_orbits,
            contribution_components=contribution_components,
            orbits=tuple(specifications),
            labels=tuple(labels),
            initial_raw_values=np.asarray(raw_values, dtype=np.float64),
            probability_floor=float(probability_floor),
        )

    @property
    def independent_count(self) -> int:
        return len(self.initial_raw_values)

    def initial_values(self, backend, *, requires_grad: bool = False):
        values = backend.asarray(self.initial_raw_values, dtype=backend.dtype)
        if getattr(backend, "is_torch", False):
            values = values.clone().detach().requires_grad_(requires_grad)
        elif requires_grad:
            raise TypeError("requires_grad is available only with TorchBackend")
        return values

    def _orbit_species_values(self, independent_values, backend):
        values = []
        for orbit in self.orbits:
            count = orbit.component_count
            if orbit.degrees_of_freedom:
                non_reference = [
                    index for index in range(count) if index != orbit.reference_component
                ]
                design = np.zeros((count, orbit.degrees_of_freedom), dtype=np.float64)
                design[non_reference, np.arange(orbit.degrees_of_freedom)] = 1.0
                logits = backend.matmul(
                    backend.asarray(design, dtype=backend.dtype),
                    independent_values[orbit.parameter_slice],
                )
                probabilities = backend.softmax(logits, axis=0)
            else:
                probabilities = backend.ones((1,), dtype=backend.dtype)
            species = probabilities[: orbit.species_count]
            if not orbit.includes_vacancy:
                species = species * orbit.total_occupancy
            values.append(species)
        return values

    def expand(self, independent_values, backend):
        if tuple(independent_values.shape) != (self.independent_count,):
            raise ValueError(
                f"independent_values must have shape ({self.independent_count},), "
                f"got {tuple(independent_values.shape)}"
            )
        orbit_values = self._orbit_species_values(independent_values, backend)
        return backend.stack(
            [
                orbit_values[int(orbit)][int(component)]
                for orbit, component in zip(self.contribution_orbits, self.contribution_components)
            ]
        )

    def physical_groups(self, independent_values) -> tuple[dict[str, object], ...]:
        from .backends import NumpyBackend

        orbit_values = self._orbit_species_values(
            np.asarray(independent_values, dtype=np.float64), NumpyBackend()
        )
        result = []
        for orbit, species_values in zip(self.orbits, orbit_values):
            species = {symbol: float(value) for symbol, value in zip(orbit.symbols, species_values)}
            result.append(
                {
                    "orbit": orbit.orbit_index,
                    "representative_site": orbit.representative_site,
                    "members": orbit.member_sites.tolist(),
                    "species": species,
                    "vacancy": float(max(0.0, 1.0 - sum(species.values()))),
                }
            )
        return tuple(result)


@dataclass(frozen=True)
class IsotropicDisplacementOrbitSpec:
    orbit_index: int
    representative_site: int
    member_sites: np.ndarray
    parameter_index: int
    label: str


@dataclass(frozen=True)
class SymmetryIsotropicDisplacementParameterization:
    """Positive Biso values shared by all sites and species in an orbit."""

    contribution_orbits: np.ndarray
    orbits: tuple[IsotropicDisplacementOrbitSpec, ...]
    labels: tuple[str, ...]
    initial_raw_values: np.ndarray
    b_min: float

    @classmethod
    def from_calculator(
        cls,
        calculator,
        *,
        b_min: float = 0.0,
        default_if_zero: float = 0.5,
    ) -> "SymmetryIsotropicDisplacementParameterization":
        calculator._ensure_loaded()
        if not np.isfinite(b_min) or b_min < 0:
            raise ValueError("b_min must be finite and non-negative")
        if not np.isfinite(default_if_zero) or default_if_zero <= b_min:
            raise ValueError("default_if_zero must be finite and greater than b_min")
        metadata = calculator._symm
        site_indices = np.asarray(metadata["site_indices"], dtype=np.int64)
        b_iso = np.asarray(metadata["B"], dtype=np.float64)
        symbols = np.asarray(metadata["symbols"], dtype=object)
        orbit_members = tuple(
            np.asarray(item, dtype=np.int64) for item in metadata["orbit_indices"]
        )
        contribution_orbits = np.full(len(b_iso), -1, dtype=np.int64)
        specifications = []
        raw_values = []
        labels = []
        for orbit_index, members in enumerate(orbit_members):
            representative = int(members[0])
            representative_contributions = np.flatnonzero(site_indices == representative)
            values = b_iso[representative_contributions]
            if len(values) == 0:
                raise RuntimeError(f"orbit {orbit_index} has no scattering contributions")
            if np.ptp(values) > 1e-8:
                raise ValueError(f"orbit {orbit_index} has species-dependent Biso values")
            initial_b = float(values[0]) if values[0] > b_min + 1e-10 else default_if_zero
            positive = max(initial_b - b_min, 1e-12)
            raw_values.append(float(positive if positive > 20.0 else np.log(np.expm1(positive))))
            symbol_label = "/".join(
                dict.fromkeys(str(symbols[index]) for index in representative_contributions)
            )
            label = f"orbit_{orbit_index}.{symbol_label}.B_iso"
            labels.append(label)
            specifications.append(
                IsotropicDisplacementOrbitSpec(
                    orbit_index=orbit_index,
                    representative_site=representative,
                    member_sites=members.copy(),
                    parameter_index=orbit_index,
                    label=label,
                )
            )
            for member in members:
                contributions = np.flatnonzero(site_indices == member)
                if np.ptp(b_iso[contributions]) > 1e-8:
                    raise ValueError(f"site {member} has species-dependent Biso values")
                contribution_orbits[contributions] = orbit_index
        if np.any(contribution_orbits < 0):
            raise RuntimeError("not every scattering contribution was assigned to a Biso orbit")
        return cls(
            contribution_orbits=contribution_orbits,
            orbits=tuple(specifications),
            labels=tuple(labels),
            initial_raw_values=np.asarray(raw_values, dtype=np.float64),
            b_min=float(b_min),
        )

    @property
    def independent_count(self) -> int:
        return len(self.orbits)

    def initial_values(self, backend, *, requires_grad: bool = False):
        values = backend.asarray(self.initial_raw_values, dtype=backend.dtype)
        if getattr(backend, "is_torch", False):
            values = values.clone().detach().requires_grad_(requires_grad)
        elif requires_grad:
            raise TypeError("requires_grad is available only with TorchBackend")
        return values

    def orbit_values(self, independent_values, backend):
        return self.b_min + backend.softplus(independent_values)

    def expand(self, independent_values, backend):
        if tuple(independent_values.shape) != (self.independent_count,):
            raise ValueError(
                f"independent_values must have shape ({self.independent_count},), "
                f"got {tuple(independent_values.shape)}"
            )
        values = self.orbit_values(independent_values, backend)
        indices = backend.asarray(self.contribution_orbits, dtype=backend.int64)
        return values[indices]

    def physical_groups(self, independent_values) -> tuple[dict[str, object], ...]:
        from .backends import NumpyBackend

        values = self.orbit_values(np.asarray(independent_values, dtype=np.float64), NumpyBackend())
        return tuple(
            {
                "orbit": orbit.orbit_index,
                "representative_site": orbit.representative_site,
                "members": orbit.member_sites.tolist(),
                "B_iso": float(values[orbit.parameter_index]),
            }
            for orbit in self.orbits
        )


@dataclass(frozen=True)
class OrbitCoordinateSpec:
    """Independent displacement subspace and member mappings for one orbit."""

    orbit_index: int
    representative_site: int
    member_sites: np.ndarray
    basis: np.ndarray
    member_rotations: np.ndarray
    parameter_slice: slice

    @property
    def degrees_of_freedom(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True)
class SymmetryCoordinateParameterization:
    """Linear local coordinates that preserve the prepared Wyckoff orbits.

    The reflection topology and orbit membership remain fixed. Independent
    values are displacements from the loaded structure, not wrapped fractional
    coordinates. Site-stabilizer null spaces keep special-position constraints,
    while assigned symmetry rotations propagate every displacement to all
    members of an orbit.
    """

    base_site_coordinates: np.ndarray
    contribution_site_indices: np.ndarray
    design_matrix: np.ndarray
    orbits: tuple[OrbitCoordinateSpec, ...]
    labels: tuple[str, ...]
    symmetry_tolerance: float

    @classmethod
    def from_calculator(
        cls,
        calculator,
        *,
        symmetry_tolerance: float | None = None,
    ) -> "SymmetryCoordinateParameterization":
        calculator._ensure_loaded()
        tolerance = (
            max(1e-7, 1.1 * calculator.symprec)
            if symmetry_tolerance is None
            else float(symmetry_tolerance)
        )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("symmetry_tolerance must be positive and finite")
        metadata = calculator._symm
        base = np.asarray(metadata["structure"].frac_coords, dtype=np.float64)
        rotations = np.asarray(metadata["symm_rot"], dtype=np.int64)
        translations = np.asarray(metadata["symm_trans"], dtype=np.float64)
        orbit_members = tuple(
            np.asarray(orbit, dtype=np.int64) for orbit in metadata["orbit_indices"]
        )

        site_blocks = []
        specifications = []
        labels = []
        parameter_count = 0
        for orbit_index, members in enumerate(orbit_members):
            representative = int(members[0])
            representative_coordinate = base[representative]
            mapped_representative = (
                np.einsum("rij,j->ri", rotations, representative_coordinate) + translations
            )
            stabilizer_mask = (
                np.max(
                    np.abs(_periodic_difference(mapped_representative, representative_coordinate)),
                    axis=1,
                )
                < tolerance
            )
            stabilizer = rotations[stabilizer_mask] - np.eye(3, dtype=np.int64)
            basis = _null_space(stabilizer.reshape(-1, 3), tolerance)

            member_rotations = []
            for member in members:
                errors = np.max(
                    np.abs(_periodic_difference(mapped_representative, base[member])), axis=1
                )
                operation_index = int(np.argmin(errors))
                if errors[operation_index] >= tolerance:
                    raise ValueError(
                        f"could not map orbit {orbit_index} representative to site {member}"
                    )
                member_rotations.append(rotations[operation_index])

            start = parameter_count
            parameter_count += basis.shape[1]
            parameter_slice = slice(start, parameter_count)
            specifications.append(
                OrbitCoordinateSpec(
                    orbit_index=orbit_index,
                    representative_site=representative,
                    member_sites=members.copy(),
                    basis=basis,
                    member_rotations=np.asarray(member_rotations, dtype=np.int64),
                    parameter_slice=parameter_slice,
                )
            )
            labels.extend(f"orbit_{orbit_index}.u_{index}" for index in range(basis.shape[1]))
            for member, rotation in zip(members, member_rotations):
                site_blocks.append((int(member), parameter_slice, rotation @ basis))

        site_design = np.zeros((len(base), 3, parameter_count), dtype=np.float64)
        for site, parameter_slice, block in site_blocks:
            site_design[site, :, parameter_slice] = block
        contribution_indices = np.asarray(metadata["site_indices"], dtype=np.int64)
        contribution_design = site_design[contribution_indices].reshape(
            len(contribution_indices) * 3, parameter_count
        )
        return cls(
            base_site_coordinates=base,
            contribution_site_indices=contribution_indices,
            design_matrix=contribution_design,
            orbits=tuple(specifications),
            labels=tuple(labels),
            symmetry_tolerance=tolerance,
        )

    @property
    def independent_count(self) -> int:
        return self.design_matrix.shape[1]

    def initial_values(self, backend, *, requires_grad: bool = False):
        """Return a zero displacement vector on the configured backend."""
        values = backend.zeros((self.independent_count,), dtype=backend.dtype)
        if getattr(backend, "is_torch", False):
            values = values.clone().detach().requires_grad_(requires_grad)
        elif requires_grad:
            raise TypeError("requires_grad is available only with TorchBackend")
        return values

    def expand(self, independent_values, backend):
        """Expand independent displacements to all scattering contributions."""
        if tuple(independent_values.shape) != (self.independent_count,):
            raise ValueError(
                f"independent_values must have shape ({self.independent_count},), "
                f"got {tuple(independent_values.shape)}"
            )
        base_contributions = self.base_site_coordinates[self.contribution_site_indices]
        base = backend.asarray(base_contributions.reshape(-1), dtype=backend.dtype)
        design = backend.asarray(self.design_matrix, dtype=backend.dtype)
        expanded = base + backend.matmul(design, independent_values)
        return expanded.reshape(len(self.contribution_site_indices), 3)

    def forward_parameters(self, calculator, independent_values, *, base_parameters=None):
        """Build a calculator parameter dictionary with expanded coordinates."""
        parameters = (
            calculator.tensor_parameters() if base_parameters is None else dict(base_parameters)
        )
        parameters["frac_coords"] = self.expand(independent_values, calculator.backend)
        return parameters
