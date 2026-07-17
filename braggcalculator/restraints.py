"""Differentiable structural restraints with fixed crystallographic topology."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np


def _positive_finite(name, value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _image(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("periodic image must be a finite length-three vector")
    if not np.allclose(result, np.rint(result), atol=1e-12):
        raise ValueError("periodic image must contain integers")
    return np.rint(result).astype(np.int64)


@dataclass(frozen=True)
class CompositionRestraint:
    species: str
    target: float
    sigma: float

    def __post_init__(self):
        if not self.species:
            raise ValueError("composition species cannot be empty")
        if not np.isfinite(self.target) or self.target < 0:
            raise ValueError("composition target must be finite and non-negative")
        object.__setattr__(self, "sigma", _positive_finite("composition sigma", self.sigma))


@dataclass(frozen=True)
class BondLengthRestraint:
    sites: tuple[int, int]
    target: float
    sigma: float
    image: np.ndarray

    def __post_init__(self):
        if len(self.sites) != 2 or self.sites[0] == self.sites[1]:
            raise ValueError("bond restraint requires two distinct site indices")
        object.__setattr__(self, "target", _positive_finite("bond target", self.target))
        object.__setattr__(self, "sigma", _positive_finite("bond sigma", self.sigma))
        object.__setattr__(self, "image", _image(self.image))


@dataclass(frozen=True)
class BondAngleRestraint:
    sites: tuple[int, int, int]
    target_degrees: float
    sigma_degrees: float
    outer_images: tuple[np.ndarray, np.ndarray]

    def __post_init__(self):
        if len(self.sites) != 3 or len(set(self.sites)) != 3:
            raise ValueError("angle restraint requires three distinct sites (outer, center, outer)")
        if not np.isfinite(self.target_degrees) or not 0 < self.target_degrees < 180:
            raise ValueError("angle target must lie strictly between 0 and 180 degrees")
        object.__setattr__(
            self,
            "sigma_degrees",
            _positive_finite("angle sigma", self.sigma_degrees),
        )
        if len(self.outer_images) != 2:
            raise ValueError("outer_images must contain two periodic images")
        object.__setattr__(
            self,
            "outer_images",
            tuple(_image(value) for value in self.outer_images),
        )


@dataclass(frozen=True)
class MinimumDistanceRestraint:
    sites: tuple[int, int]
    minimum: float
    sigma: float
    image: np.ndarray

    def __post_init__(self):
        if len(self.sites) != 2 or self.sites[0] == self.sites[1]:
            raise ValueError("minimum-distance restraint requires two distinct sites")
        object.__setattr__(self, "minimum", _positive_finite("minimum distance", self.minimum))
        object.__setattr__(self, "sigma", _positive_finite("distance sigma", self.sigma))
        object.__setattr__(self, "image", _image(self.image))


def _nearest_image(frac_left, frac_right, lattice):
    delta = np.asarray(frac_right) - np.asarray(frac_left)
    center = -np.rint(delta).astype(np.int64)
    candidates = np.asarray(
        [center + np.asarray(offset) for offset in product((-1, 0, 1), repeat=3)]
    )
    cartesian = (delta[None, :] + candidates) @ np.asarray(lattice)
    return candidates[int(np.argmin(np.sum(cartesian**2, axis=1)))]


@dataclass(frozen=True)
class StructuralRestraintSet:
    """Resolved restraints whose periodic images remain fixed during refinement."""

    site_contributions: np.ndarray
    contribution_symbols: tuple[str, ...]
    composition: tuple[CompositionRestraint, ...] = ()
    bonds: tuple[BondLengthRestraint, ...] = ()
    angles: tuple[BondAngleRestraint, ...] = ()
    minimum_distances: tuple[MinimumDistanceRestraint, ...] = ()

    @property
    def count(self):
        return (
            len(self.composition) + len(self.bonds) + len(self.angles) + len(self.minimum_distances)
        )

    @classmethod
    def from_dict(cls, calculator, specification: Mapping[str, Any] | None):
        calculator._ensure_loaded()
        specification = {} if specification is None else dict(specification)
        allowed = {"composition", "bonds", "angles", "minimum_distances"}
        unknown = set(specification) - allowed
        if unknown:
            raise ValueError(f"unknown structural restraint groups: {sorted(unknown)}")
        metadata = calculator._symm
        structure = metadata["structure"]
        coordinates = np.asarray(structure.frac_coords, dtype=np.float64)
        lattice = np.asarray(metadata["lattice"], dtype=np.float64)
        site_indices = np.asarray(metadata["site_indices"], dtype=np.int64)
        site_contributions = np.asarray(
            [np.flatnonzero(site_indices == site)[0] for site in range(len(structure))],
            dtype=np.int64,
        )

        def validate_sites(values, expected):
            sites = tuple(int(value) for value in values)
            if len(sites) != expected or any(site < 0 or site >= len(structure) for site in sites):
                raise ValueError(f"restraint sites must contain {expected} valid site indices")
            return sites

        composition = tuple(
            CompositionRestraint(
                species=str(item["species"]),
                target=float(item["target"]),
                sigma=float(item["sigma"]),
            )
            for item in specification.get("composition", ())
        )
        bonds = []
        for item in specification.get("bonds", ()):
            sites = validate_sites(item["sites"], 2)
            image = item.get("image")
            if image is None:
                image = _nearest_image(coordinates[sites[0]], coordinates[sites[1]], lattice)
            bonds.append(
                BondLengthRestraint(
                    sites=sites,
                    target=float(item["target"]),
                    sigma=float(item["sigma"]),
                    image=image,
                )
            )
        angles = []
        for item in specification.get("angles", ()):
            sites = validate_sites(item["sites"], 3)
            images = item.get("outer_images")
            if images is None:
                images = (
                    _nearest_image(coordinates[sites[1]], coordinates[sites[0]], lattice),
                    _nearest_image(coordinates[sites[1]], coordinates[sites[2]], lattice),
                )
            angles.append(
                BondAngleRestraint(
                    sites=sites,
                    target_degrees=float(item["target_degrees"]),
                    sigma_degrees=float(item["sigma_degrees"]),
                    outer_images=images,
                )
            )
        minimum_distances = []
        for item in specification.get("minimum_distances", ()):
            sites = validate_sites(item["sites"], 2)
            image = item.get("image")
            if image is None:
                image = _nearest_image(coordinates[sites[0]], coordinates[sites[1]], lattice)
            minimum_distances.append(
                MinimumDistanceRestraint(
                    sites=sites,
                    minimum=float(item["minimum"]),
                    sigma=float(item["sigma"]),
                    image=image,
                )
            )
        return cls(
            site_contributions=site_contributions,
            contribution_symbols=tuple(str(item) for item in metadata["symbols"]),
            composition=composition,
            bonds=tuple(bonds),
            angles=tuple(angles),
            minimum_distances=tuple(minimum_distances),
        )

    def _site_coordinates(self, frac_coords):
        return frac_coords[self.site_contributions]

    @staticmethod
    def _distance(site_coordinates, lattice, sites, image, backend):
        delta = site_coordinates[sites[1]] - site_coordinates[sites[0]]
        delta = delta + backend.asarray(image, dtype=backend.dtype)
        cartesian = backend.matmul(delta, lattice)
        return backend.sqrt(backend.sum(cartesian**2))

    def residuals(self, lattice, frac_coords, occupancies, backend):
        """Return named standardized restraint residuals before squaring."""
        if self.count == 0:
            return {}
        site_coordinates = self._site_coordinates(frac_coords)
        residuals = {}
        for index, restraint in enumerate(self.composition):
            selected = [
                position
                for position, symbol in enumerate(self.contribution_symbols)
                if symbol == restraint.species
            ]
            if not selected:
                raise ValueError(f"composition species {restraint.species!r} is absent")
            value = backend.sum(occupancies[backend.asarray(selected, dtype=backend.int64)])
            residuals[f"composition[{index}].{restraint.species}"] = (
                value - restraint.target
            ) / restraint.sigma
        for index, restraint in enumerate(self.bonds):
            value = self._distance(
                site_coordinates,
                lattice,
                restraint.sites,
                restraint.image,
                backend,
            )
            residuals[f"bond[{index}]"] = (value - restraint.target) / restraint.sigma
        for index, restraint in enumerate(self.angles):
            center = site_coordinates[restraint.sites[1]]
            vectors = []
            for site, image in zip(
                (restraint.sites[0], restraint.sites[2]), restraint.outer_images
            ):
                delta = site_coordinates[site] - center
                delta = delta + backend.asarray(image, dtype=backend.dtype)
                vectors.append(backend.matmul(delta, lattice))
            denominator = backend.sqrt(backend.sum(vectors[0] ** 2)) * backend.sqrt(
                backend.sum(vectors[1] ** 2)
            )
            cosine = backend.sum(vectors[0] * vectors[1]) / denominator
            angle = backend.arccos(backend.clip(cosine, -1.0 + 1e-12, 1.0 - 1e-12))
            target = np.radians(restraint.target_degrees)
            sigma = np.radians(restraint.sigma_degrees)
            residuals[f"angle[{index}]"] = (angle - target) / sigma
        for index, restraint in enumerate(self.minimum_distances):
            value = self._distance(
                site_coordinates,
                lattice,
                restraint.sites,
                restraint.image,
                backend,
            )
            violation = (restraint.minimum - value) / restraint.sigma
            zero = violation * 0.0
            penalty = backend.where(violation > 0, violation, zero)
            residuals[f"minimum_distance[{index}]"] = penalty
        return residuals

    def loss(self, lattice, frac_coords, occupancies, backend):
        """Return mean standardized penalty and named squared contributions."""
        residuals = self.residuals(lattice, frac_coords, occupancies, backend)
        if not residuals:
            zero = backend.sum(frac_coords) * 0.0
            return zero, {}
        contributions = {name: value**2 for name, value in residuals.items()}
        terms = list(contributions.values())
        return backend.sum(backend.stack(terms)) / len(terms), contributions

    def specification(self):
        return {
            "composition": [
                {"species": item.species, "target": item.target, "sigma": item.sigma}
                for item in self.composition
            ],
            "bonds": [
                {
                    "sites": list(item.sites),
                    "target": item.target,
                    "sigma": item.sigma,
                    "image": item.image.tolist(),
                }
                for item in self.bonds
            ],
            "angles": [
                {
                    "sites": list(item.sites),
                    "target_degrees": item.target_degrees,
                    "sigma_degrees": item.sigma_degrees,
                    "outer_images": [value.tolist() for value in item.outer_images],
                }
                for item in self.angles
            ],
            "minimum_distances": [
                {
                    "sites": list(item.sites),
                    "minimum": item.minimum,
                    "sigma": item.sigma,
                    "image": item.image.tolist(),
                }
                for item in self.minimum_distances
            ],
        }
