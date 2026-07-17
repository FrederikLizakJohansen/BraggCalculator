"""Experimental powder-diffraction datasets and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class DiffractionDataset:
    """One measured powder pattern with uncertainty, masks, and metadata."""

    coordinate: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    domain: Literal["two_theta", "q"]
    wavelength: float
    radiation: Literal["xray", "neutron"] = "xray"
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: str | None = None
    source_sha256: str | None = None
    observation_covariance: np.ndarray | None = None

    def __post_init__(self):
        coordinate = np.asarray(self.coordinate, dtype=np.float64)
        intensity = np.asarray(self.intensity, dtype=np.float64)
        sigma = np.asarray(self.sigma, dtype=np.float64)
        mask = np.asarray(self.mask, dtype=bool)
        shape = coordinate.shape
        if coordinate.ndim != 1 or len(coordinate) < 3:
            raise ValueError("dataset must contain at least three one-dimensional points")
        if intensity.shape != shape or sigma.shape != shape or mask.shape != shape:
            raise ValueError("coordinate, intensity, sigma, and mask must have equal shapes")
        if not np.all(np.isfinite(coordinate)) or np.any(np.diff(coordinate) <= 0):
            raise ValueError("coordinate must be finite and strictly increasing")
        if not np.all(np.isfinite(intensity)):
            raise ValueError("intensity must be finite")
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("sigma must be positive and finite")
        covariance = self.observation_covariance
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            expected_shape = (len(coordinate), len(coordinate))
            if covariance.shape != expected_shape or not np.all(np.isfinite(covariance)):
                raise ValueError(
                    f"observation_covariance must be a finite matrix with shape {expected_shape}"
                )
            if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
                raise ValueError("observation_covariance must be symmetric")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as error:
                raise ValueError("observation_covariance must be positive definite") from error
            if not np.allclose(np.diag(covariance), sigma**2, rtol=1e-6, atol=1e-12):
                raise ValueError("sigma squared must match the diagonal of observation_covariance")
        if not np.any(mask):
            raise ValueError("dataset mask excludes every observation")
        if self.domain not in {"two_theta", "q"}:
            raise ValueError("domain must be 'two_theta' or 'q'")
        if self.radiation not in {"xray", "neutron"}:
            raise ValueError("radiation must be 'xray' or 'neutron'")
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "observation_covariance", covariance)

    @classmethod
    def from_xye(
        cls,
        path,
        *,
        domain: Literal["two_theta", "q"] = "two_theta",
        wavelength: float,
        radiation: Literal["xray", "neutron"] = "xray",
        third_column: Literal["sigma", "weight"] = "sigma",
        metadata: Mapping[str, object] | None = None,
    ) -> "DiffractionDataset":
        """Read whitespace/comma-separated x, y, sigma-or-weight data."""
        source_path = Path(path)
        content = source_path.read_bytes()
        first_data_line = next(
            (line for line in content.decode("utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")),
            "",
        )
        rows = np.genfromtxt(
            source_path, comments="#", delimiter="," if "," in first_data_line else None
        )
        if rows.ndim != 2 or rows.shape[1] < 2:
            raise ValueError("experimental file must contain at least x and intensity columns")
        coordinate = rows[:, 0]
        intensity = rows[:, 1]
        if rows.shape[1] >= 3:
            third = rows[:, 2]
            sigma = third if third_column == "sigma" else 1.0 / np.sqrt(third)
        else:
            sigma = np.sqrt(np.maximum(intensity, 0.0) + 1.0)
        return cls(
            coordinate=coordinate,
            intensity=intensity,
            sigma=sigma,
            mask=np.ones(len(coordinate), dtype=bool),
            domain=domain,
            wavelength=wavelength,
            radiation=radiation,
            metadata={} if metadata is None else metadata,
            source=str(source_path),
            source_sha256=sha256(content).hexdigest(),
        )

    def select_range(self, lower: float, upper: float) -> "DiffractionDataset":
        """Return a cropped immutable dataset."""
        selected = (self.coordinate >= lower) & (self.coordinate <= upper)
        if np.count_nonzero(selected) < 3:
            raise ValueError("selected range contains fewer than three points")
        return replace(
            self,
            coordinate=self.coordinate[selected],
            intensity=self.intensity[selected],
            sigma=self.sigma[selected],
            mask=self.mask[selected],
            observation_covariance=(
                self.observation_covariance[np.ix_(selected, selected)]
                if self.observation_covariance is not None
                else None
            ),
        )

    def exclude(self, ranges) -> "DiffractionDataset":
        """Return a dataset with specified closed coordinate ranges masked."""
        mask = self.mask.copy()
        for lower, upper in ranges:
            mask &= ~((self.coordinate >= lower) & (self.coordinate <= upper))
        return replace(self, mask=mask)

    @property
    def step(self) -> float:
        return float(np.median(np.diff(self.coordinate)))

    @property
    def weights(self) -> np.ndarray:
        return np.where(self.mask, 1.0 / self.sigma**2, 0.0)

    @property
    def observation_covariance_sha256(self) -> str | None:
        if self.observation_covariance is None:
            return None
        content = np.ascontiguousarray(self.observation_covariance).tobytes()
        return sha256(content).hexdigest()
