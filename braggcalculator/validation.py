"""Reference-validation records, gates, and independent line-pattern checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from .core import BraggCalculator
from .dataset import DiffractionDataset


ValidationStatus = Literal["pass", "warn", "fail", "unsupported", "pending_review"]
MetricDirection = Literal["maximum", "minimum", "informational"]

_SEVERITY = {"pass": 0, "warn": 1, "pending_review": 2, "unsupported": 3, "fail": 4}


@dataclass(frozen=True)
class ReferenceSource:
    """One immutable external input and its scientific provenance."""

    identifier: str
    relative_path: str
    sha256: str
    source_url: str
    material: str
    instrument: str
    radiation: Literal["xray", "neutron"]
    wavelength_angstrom: float
    data_format: str
    expected_points: int
    notes: str = ""

    def verify(self, root: Path) -> bool:
        content = (root / self.relative_path).read_bytes()
        return sha256(content).hexdigest() == self.sha256


@dataclass(frozen=True)
class ValidationMetric:
    """A numerical result with an executable pass/warn/fail gate."""

    name: str
    value: float | None
    unit: str = ""
    direction: MetricDirection = "informational"
    pass_limit: float | None = None
    warn_limit: float | None = None
    declared_status: ValidationStatus | None = None
    explanation: str = ""

    @property
    def status(self) -> ValidationStatus:
        if self.declared_status is not None:
            return self.declared_status
        if self.direction == "informational":
            return "pass"
        if self.value is None or not np.isfinite(self.value):
            return "fail"
        if self.pass_limit is None or self.warn_limit is None:
            raise ValueError("gated metrics require pass_limit and warn_limit")
        if self.direction == "maximum":
            return "pass" if self.value <= self.pass_limit else (
                "warn" if self.value <= self.warn_limit else "fail"
            )
        return "pass" if self.value >= self.pass_limit else (
            "warn" if self.value >= self.warn_limit else "fail"
        )


@dataclass(frozen=True)
class ValidationCase:
    """A named validation claim whose evidence cannot hide failed metrics."""

    identifier: str
    category: str
    description: str
    metrics: tuple[ValidationMetric, ...] = ()
    source_identifiers: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    declared_status: ValidationStatus | None = None

    @property
    def status(self) -> ValidationStatus:
        statuses = [metric.status for metric in self.metrics]
        if self.declared_status is not None:
            statuses.append(self.declared_status)
        return max(statuses, key=_SEVERITY.get) if statuses else "pass"


@dataclass(frozen=True)
class ValidationMatrix:
    """Auditable collection of validation cases and required coverage."""

    cases: tuple[ValidationCase, ...]
    sources: tuple[ReferenceSource, ...] = ()
    required_categories: tuple[str, ...] = ()
    expert_review_status: ValidationStatus = "pending_review"
    expert_review_checklist: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def missing_categories(self) -> tuple[str, ...]:
        observed = {case.category for case in self.cases}
        return tuple(category for category in self.required_categories if category not in observed)

    @property
    def overall_status(self) -> ValidationStatus:
        statuses = [case.status for case in self.cases]
        if self.missing_categories:
            statuses.append("fail")
        statuses.append(self.expert_review_status)
        return max(statuses, key=_SEVERITY.get)

    @property
    def status_counts(self) -> dict[str, int]:
        return {
            status: sum(case.status == status for case in self.cases)
            for status in _SEVERITY
        }

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for case, encoded in zip(self.cases, result["cases"]):
            encoded["status"] = case.status
            for metric, metric_encoded in zip(case.metrics, encoded["metrics"]):
                metric_encoded["status"] = metric.status
        result["overall_status"] = self.overall_status
        result["missing_categories"] = self.missing_categories
        result["status_counts"] = self.status_counts
        return result

    def write_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_reference_sources(path) -> tuple[ReferenceSource, ...]:
    """Load the checked-in external-data manifest."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(ReferenceSource(**record) for record in records["sources"])


def validate_public_sources(
    root: Path, sources: Sequence[ReferenceSource]
) -> tuple[ValidationCase, ...]:
    """Verify checksums and ingest every declared constant-wavelength dataset."""
    cases = []
    for source in sources:
        path = root / source.relative_path
        checksum_matches = source.verify(root)
        reader = (
            DiffractionDataset.from_xye
            if "xye" in source.data_format.lower()
            else DiffractionDataset.from_gsas_constant_step
        )
        dataset = reader(
            path, wavelength=source.wavelength_angstrom, radiation=source.radiation,
            metadata={"reference_identifier": source.identifier},
        )
        cases.append(
            ValidationCase(
                identifier=f"public-data:{source.identifier}",
                category="public_data",
                description=f"Checksum and powder-data ingestion for {source.material}",
                source_identifiers=(source.identifier,),
                metrics=(
                    ValidationMetric(
                        "checksum_match", float(checksum_matches), direction="minimum",
                        pass_limit=1.0, warn_limit=1.0,
                    ),
                    ValidationMetric(
                        "point_count", float(len(dataset.coordinate)), "points",
                        direction="minimum", pass_limit=float(source.expected_points),
                        warn_limit=float(source.expected_points),
                    ),
                    ValidationMetric(
                        "finite_fraction",
                        float(np.mean(np.isfinite(dataset.intensity))),
                        direction="minimum", pass_limit=1.0, warn_limit=0.999,
                    ),
                ),
                assumptions=(
                    str(
                        dataset.metadata.get(
                            "uncertainty_interpretation",
                            "Uncertainty interpretation is documented in the vendored XYE header",
                        )
                    ),
                ),
            )
        )
    return tuple(cases)


def validate_line_oracle(
    structures: Mapping[str, object],
    *,
    modes: Sequence[Literal["xray", "neutron"]] = ("xray", "neutron"),
    position_pass: float = 1e-10,
    intensity_pass: float = 1e-9,
) -> tuple[ValidationCase, ...]:
    """Compare line positions and scaled intensities with pymatgen calculators."""
    cases = []
    for mode in modes:
        oracle_type = XRDCalculator if mode == "xray" else NDCalculator
        for name, structure in structures.items():
            calculator = BraggCalculator(mode=mode).load(structure)
            actual_x, actual_y = calculator.line_pattern(scaled=True)
            oracle = oracle_type(wavelength=calculator.wavelength).get_pattern(
                structure, two_theta_range=calculator.two_theta_range, scaled=True
            )
            same_count = len(actual_x) == len(oracle.x)
            if same_count:
                position_error = float(np.max(np.abs(np.asarray(actual_x) - oracle.x), initial=0))
                intensity_error = float(np.max(np.abs(np.asarray(actual_y) - oracle.y), initial=0))
            else:
                position_error = intensity_error = float("inf")
            cases.append(
                ValidationCase(
                    identifier=f"line-oracle:{mode}:{name}",
                    category="line_oracle",
                    description=f"{mode} line pattern against pymatgen for {name}",
                    metrics=(
                        ValidationMetric(
                            "peak_count_delta", float(abs(len(actual_x) - len(oracle.x))),
                            "peaks", "maximum", 0.0, 0.0,
                        ),
                        ValidationMetric(
                            "maximum_position_error", position_error, "degrees", "maximum",
                            position_pass, position_pass * 100,
                        ),
                        ValidationMetric(
                            "maximum_scaled_intensity_error", intensity_error, "%", "maximum",
                            intensity_pass, intensity_pass * 100,
                        ),
                    ),
                    assumptions=("pymatgen is an independent line-pattern oracle, not a profile-refinement oracle",),
                )
            )
    return tuple(cases)
