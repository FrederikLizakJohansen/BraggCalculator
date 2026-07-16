"""Deterministic structures for atom-count and symmetry scaling studies."""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.core import Lattice, Structure


@dataclass(frozen=True)
class ScalingCase:
    """One structure and the controlled variable it represents."""

    name: str
    series: str
    structure: Structure
    control_value: int


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def p1_structure(site_count: int) -> Structure:
    """Build a fixed-density triclinic P1 cell from a low-discrepancy sequence."""
    if site_count < 2:
        raise ValueError("P1 scaling structures require at least two sites")
    scale = (site_count / 4.0) ** (1.0 / 3.0)
    lattice = Lattice.from_parameters(
        5.1 * scale,
        5.6 * scale,
        6.2 * scale,
        78.0,
        83.0,
        74.0,
    )
    coordinates = [
        [_radical_inverse(index, base) for base in (2, 3, 5)]
        for index in range(1, site_count + 1)
    ]
    species_cycle = ("Si", "O", "Al", "Ca", "Na")
    species = [species_cycle[index % len(species_cycle)] for index in range(site_count)]
    return Structure(lattice, species, coordinates)


def nacl_supercell(factor: int) -> Structure:
    """Build a conventional NaCl supercell with a two-site primitive cell."""
    if factor < 1:
        raise ValueError("NaCl supercell factors must be positive")
    structure = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.6402),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    structure.make_supercell((factor, factor, factor))
    return structure


def scaling_cases(
    p1_site_counts: tuple[int, ...],
    symmetry_factors: tuple[int, ...],
) -> list[ScalingCase]:
    """Return both controlled benchmark series in increasing input size."""
    if tuple(sorted(set(p1_site_counts))) != p1_site_counts:
        raise ValueError("P1 site counts must be unique and strictly increasing")
    if tuple(sorted(set(symmetry_factors))) != symmetry_factors:
        raise ValueError("symmetry factors must be unique and strictly increasing")
    cases = [
        ScalingCase(f"P1-{sites}", "p1", p1_structure(sites), sites)
        for sites in p1_site_counts
    ]
    cases.extend(
        ScalingCase(
            f"NaCl-{factor}x{factor}x{factor}",
            "symmetry",
            nacl_supercell(factor),
            factor,
        )
        for factor in symmetry_factors
    )
    return cases
