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
            cartesian_rotation = (
                lattice.T @ fractional_rotation @ inverse_cartesian_lattice
            )
            transformed = np.asarray(
                [cartesian_rotation @ item @ cartesian_rotation.T for item in matrix_basis]
            )
            transform = np.stack(
                [_symmetric_coefficients(item) for item in transformed], axis=1
            )
            constraints.append(transform - np.eye(6))
        constraint_matrix = np.concatenate(constraints, axis=0)
        coefficients = _null_space(constraint_matrix, tolerance)
        strain_basis = np.einsum("ki,kjl->ijl", coefficients, matrix_basis)
        crystal_system = str(metadata["crystal_system"])
        labels = tuple(
            f"{crystal_system}.metric_mode_{index + 1}"
            for index in range(strain_basis.shape[0])
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
        log_strain = self.strain_scale * backend.einsum(
            "k,kij->ij", independent_values, basis
        )
        deformation = backend.matrix_exp(log_strain)
        base = backend.asarray(self.base_lattice, dtype=backend.dtype)
        return backend.matmul(base, deformation)

    def physical_parameters(self, independent_values) -> dict[str, float]:
        from .backends import NumpyBackend

        values = np.asarray(independent_values, dtype=np.float64)
        lattice = self.expand(values, NumpyBackend())
        return lattice_parameters(lattice)


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
        orbit_members = tuple(np.asarray(orbit, dtype=np.int64) for orbit in metadata["orbit_indices"])

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
            stabilizer_mask = np.max(
                np.abs(_periodic_difference(mapped_representative, representative_coordinate)),
                axis=1,
            ) < tolerance
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
            calculator.tensor_parameters()
            if base_parameters is None
            else dict(base_parameters)
        )
        parameters["frac_coords"] = self.expand(independent_values, calculator.backend)
        return parameters
