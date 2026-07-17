"""Versioned, portable refinement projects and scientific export bundles."""

from __future__ import annotations

import csv
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from .core import BraggCalculator
from .dataset import DiffractionDataset
from .optimization import OptimizationStage
from .session import RefinementPolicy, RefinementSession, SessionResult


PROJECT_SCHEMA = "braggcalculator.project/v1"
RESULT_SCHEMA = "braggcalculator.session-result/v1"
AUDIT_SCHEMA = "braggcalculator.audit/v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def policy_to_dict(policy: RefinementPolicy) -> dict[str, Any]:
    """Serialize a refinement policy without losing stage definitions."""
    return _jsonable(asdict(policy))


def policy_from_dict(value: Mapping[str, Any]) -> RefinementPolicy:
    """Load a policy while rejecting unknown fields and malformed stages."""
    supplied = dict(value)
    allowed = {item.name for item in fields(RefinementPolicy)}
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"unknown refinement policy fields: {sorted(unknown)}")
    if supplied.get("stages") is not None:
        supplied["stages"] = tuple(
            OptimizationStage(
                item["name"], tuple(item["active"]), int(item["steps"]),
                float(item["learning_rate"]), optimizer=item.get("optimizer", "adam"),
                width_multiplier=float(item.get("width_multiplier", 1.0)),
            )
            for item in supplied["stages"]
        )
    if supplied.get("rigid_bodies") is not None:
        supplied["rigid_bodies"] = tuple(supplied["rigid_bodies"])
    return RefinementPolicy(**supplied)


def validate_project_document(document: Mapping[str, Any]) -> None:
    """Apply the project invariants used by both disk and service interfaces."""
    required = {
        "schema", "project_id", "title", "created_at", "updated_at", "dataset",
        "models", "policy", "runs", "audit",
    }
    missing = required - set(document)
    if missing:
        raise ValueError(f"project is missing required fields: {sorted(missing)}")
    if document["schema"] != PROJECT_SCHEMA:
        raise ValueError(f"unsupported project schema: {document['schema']!r}")
    if not document["models"]:
        raise ValueError("project must contain at least one model")
    names = [item["name"] for item in document["models"]]
    if len(names) != len(set(names)):
        raise ValueError("project model names must be unique")
    policy_from_dict(document["policy"])


def session_result_to_dict(result: SessionResult, *, run_id: str) -> dict[str, Any]:
    """Convert a session result into the stable v1 structured result schema."""
    candidates = []
    for candidate in result.candidates:
        candidates.append(
            {
                "name": candidate.name,
                "formula": candidate.structure.composition.reduced_formula,
                "r_wp": candidate.r_wp,
                "chi_squared": candidate.chi_squared,
                "held_out_r_wp": candidate.held_out_r_wp,
                "calculated": candidate.calculated,
                "residual": candidate.residual,
                "physical_parameters": candidate.physical_parameters,
                "loss_history": candidate.loss_history,
                "stage_history": candidate.stage_history,
                "informative_regions": candidate.informative_regions,
                "identifiability": candidate.identifiability,
                "recommendation": candidate.recommendation,
                "warnings": candidate.warnings,
                "provenance": candidate.provenance,
                "convergence": candidate.convergence,
            }
        )
    return _jsonable(
        {
            "schema": RESULT_SCHEMA,
            "run_id": run_id,
            "created_at": _now(),
            "dataset": {
                "coordinate": result.dataset.coordinate,
                "intensity": result.dataset.intensity,
                "sigma": result.dataset.sigma,
                "mask": result.dataset.mask.astype(int),
                "domain": result.dataset.domain,
                "radiation": result.dataset.radiation,
                "wavelength_angstrom": result.dataset.wavelength,
                "source_sha256": result.dataset.source_sha256,
                "metadata": result.dataset.metadata,
            },
            "ranking": result.ranking,
            "pairwise_discrimination": result.pairwise_discrimination,
            "conclusion": result.conclusion,
            "candidates": candidates,
        }
    )


class ProjectStore:
    """Create, run, resume, and export one directory-backed project."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.project_path = self.directory / "project.json"

    @classmethod
    def create(
        cls,
        directory,
        *,
        dataset_path,
        model_paths: Sequence[Path | str],
        wavelength: float,
        names: Sequence[str] | None = None,
        title: str = "BraggCalculator project",
        radiation: str = "xray",
        data_format: str = "xye",
        third_column: str = "sigma",
        policy: RefinementPolicy | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ProjectStore":
        store = cls(directory)
        if store.directory.exists() and any(store.directory.iterdir()):
            raise FileExistsError(f"project directory is not empty: {store.directory}")
        store.directory.mkdir(parents=True, exist_ok=True)
        inputs = store.directory / "inputs"
        inputs.mkdir()
        dataset_source = Path(dataset_path).resolve()
        if not dataset_source.is_file():
            raise FileNotFoundError(dataset_source)
        dataset_target = inputs / f"pattern{dataset_source.suffix.lower()}"
        shutil.copy2(dataset_source, dataset_target)
        model_sources = [Path(path).resolve() for path in model_paths]
        if not model_sources:
            raise ValueError("at least one model path is required")
        for path in model_sources:
            if not path.is_file() or path.suffix.lower() != ".cif":
                raise ValueError(f"model must be an existing CIF: {path}")
        model_names = (
            tuple(names) if names is not None
            else tuple(path.stem for path in model_sources)
        )
        if len(model_names) != len(model_sources) or len(set(model_names)) != len(model_names):
            raise ValueError("model names must be unique and match model_paths")
        models = []
        for index, (name, source) in enumerate(zip(model_names, model_sources), start=1):
            target = inputs / f"model-{index:02d}.cif"
            shutil.copy2(source, target)
            models.append(
                {
                    "name": str(name),
                    "path": str(target.relative_to(store.directory)),
                    "sha256": _digest(target),
                    "original_filename": source.name,
                }
            )
        timestamp = _now()
        active_policy = RefinementPolicy.quick() if policy is None else policy
        document = {
            "schema": PROJECT_SCHEMA,
            "project_id": str(uuid4()),
            "title": str(title),
            "created_at": timestamp,
            "updated_at": timestamp,
            "dataset": {
                "path": str(dataset_target.relative_to(store.directory)),
                "sha256": _digest(dataset_target),
                "format": data_format,
                "third_column": third_column,
                "domain": "two_theta",
                "radiation": radiation,
                "wavelength_angstrom": float(wavelength),
                "metadata": {} if metadata is None else _jsonable(metadata),
            },
            "models": models,
            "policy": policy_to_dict(active_policy),
            "runs": [],
            "audit": [
                {
                    "timestamp": timestamp,
                    "action": "project_created",
                    "detail": "Input files copied and checksummed into the project bundle",
                }
            ],
        }
        store.save(document)
        return store

    def load(self) -> dict[str, Any]:
        if not self.project_path.is_file():
            raise FileNotFoundError(self.project_path)
        document = json.loads(self.project_path.read_text(encoding="utf-8"))
        validate_project_document(document)
        return document

    def save(self, document: Mapping[str, Any]) -> None:
        encoded = dict(document)
        encoded["updated_at"] = _now()
        validate_project_document(encoded)
        temporary = self.project_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(_jsonable(encoded), indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.project_path)

    def _verified_path(self, relative_path: str, expected_sha256: str) -> Path:
        path = (self.directory / relative_path).resolve()
        try:
            path.relative_to(self.directory)
        except ValueError as error:
            raise ValueError("project input path escapes the project directory") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _digest(path)
        if actual != expected_sha256:
            raise ValueError(f"project input checksum changed: {relative_path}")
        return path

    def _dataset(self, specification: Mapping[str, Any]) -> DiffractionDataset:
        path = self._verified_path(specification["path"], specification["sha256"])
        common = {
            "wavelength": float(specification["wavelength_angstrom"]),
            "radiation": specification["radiation"],
            "metadata": specification.get("metadata", {}),
        }
        if "gsas" in specification["format"].lower():
            return DiffractionDataset.from_gsas_constant_step(path, **common)
        return DiffractionDataset.from_xye(
            path, third_column=specification.get("third_column", "sigma"), **common
        )

    def _models(self, document: Mapping[str, Any]) -> tuple[list[Path], list[str]]:
        paths, names = [], []
        for model in document["models"]:
            paths.append(self._verified_path(model["path"], model["sha256"]))
            names.append(model["name"])
        return paths, names

    def run(self, *, resume: bool = False) -> tuple[dict[str, Any], SessionResult]:
        """Execute a run, optionally continuing its exact raw parameter state.

        Optimizer moments are intentionally not serialized; every continuation
        starts a fresh optimizer from the saved raw parameters and records a new
        trace segment.
        """
        document = self.load()
        dataset = self._dataset(document["dataset"])
        models, names = self._models(document)
        policy = policy_from_dict(document["policy"])
        checkpoints = None
        parent_run_id = None
        if resume:
            if not document["runs"]:
                raise ValueError("cannot resume a project without a completed run")
            parent = document["runs"][-1]
            parent_run_id = parent["run_id"]
            previous = self.read_result(parent_run_id)
            checkpoints = {
                candidate["name"]: candidate["provenance"]["checkpoint"]
                for candidate in previous["candidates"]
            }
            policy = replace(policy, restarts=1)
        session = RefinementSession(dataset, models, names=names)
        result = session.run(policy, checkpoints=checkpoints)
        run_id = f"run-{len(document['runs']) + 1:04d}"
        run_directory = self.directory / "runs" / run_id
        run_directory.mkdir(parents=True)
        encoded_result = session_result_to_dict(result, run_id=run_id)
        result_path = run_directory / "result.json"
        result_path.write_text(json.dumps(encoded_result, indent=2) + "\n", encoding="utf-8")
        artifacts = self._write_exports(result, encoded_result, run_directory)
        from .workspace import write_session_workspace

        workspace_path = write_session_workspace(
            result, run_directory / "workspace.html", project=document, run_id=run_id
        )
        artifacts["workspace_html"] = str(workspace_path.relative_to(self.directory))
        trace_segments = [
            {
                "candidate": candidate.name,
                "points": len(candidate.loss_history),
                "first_loss": float(candidate.loss_history[0]),
                "last_loss": float(candidate.loss_history[-1]),
                "resumed_from_checkpoint": bool(
                    candidate.provenance["resumed_from_checkpoint"]
                ),
            }
            for candidate in result.candidates
        ]
        record = {
            "run_id": run_id,
            "created_at": encoded_result["created_at"],
            "parent_run_id": parent_run_id,
            "resumed": resume,
            "result": str(result_path.relative_to(self.directory)),
            "ranking": list(result.ranking),
            "trace_segments": trace_segments,
            "artifacts": artifacts,
        }
        document["runs"].append(record)
        document["audit"].append(
            {
                "timestamp": _now(),
                "action": "refinement_resumed" if resume else "refinement_run",
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "detail": result.conclusion,
            }
        )
        self.save(document)
        self._write_audit(document)
        record["artifacts"]["audit_json"] = "audit.json"
        self.save(document)
        return document, result

    def read_result(self, run_id: str | None = None) -> dict[str, Any]:
        document = self.load()
        if not document["runs"]:
            raise ValueError("project has no completed runs")
        record = (
            document["runs"][-1]
            if run_id is None
            else next((item for item in document["runs"] if item["run_id"] == run_id), None)
        )
        if record is None:
            raise KeyError(f"unknown project run: {run_id}")
        return json.loads((self.directory / record["result"]).read_text(encoding="utf-8"))

    def _write_exports(
        self, result: SessionResult, encoded: Mapping[str, Any], directory: Path
    ) -> dict[str, str]:
        profile_path = directory / "profiles.csv"
        with profile_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            header = ["coordinate", "observed", "sigma", "included"]
            for candidate in result.candidates:
                header.extend([f"{candidate.name}.calculated", f"{candidate.name}.residual"])
            writer.writerow(header)
            for index in range(len(result.dataset.coordinate)):
                row = [
                    result.dataset.coordinate[index], result.dataset.intensity[index],
                    result.dataset.sigma[index], int(result.dataset.mask[index]),
                ]
                for candidate in result.candidates:
                    row.extend([candidate.calculated[index], candidate.residual[index]])
                writer.writerow(row)
        parameter_path = directory / "parameters.csv"
        with parameter_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["candidate", "parameter_path", "value"])
            for candidate in encoded["candidates"]:
                for path, value in _flatten(candidate["physical_parameters"]):
                    writer.writerow([candidate["name"], path, json.dumps(value)])
        cif_paths = {}
        for candidate in result.candidates:
            structure = refined_structure_from_candidate(candidate, result.dataset)
            path = directory / f"{_safe_name(candidate.name)}-refined.cif"
            CifWriter(structure, symprec=None).write_file(path)
            cif_paths[candidate.name] = str(path.relative_to(self.directory))
        return {
            "result_json": str((directory / "result.json").relative_to(self.directory)),
            "profiles_csv": str(profile_path.relative_to(self.directory)),
            "parameters_csv": str(parameter_path.relative_to(self.directory)),
            "refined_cif": cif_paths,
        }

    def _write_audit(self, document: Mapping[str, Any]) -> Path:
        path = self.directory / "audit.json"
        path.write_text(
            json.dumps(
                {"schema": AUDIT_SCHEMA, "project_id": document["project_id"],
                 "events": document["audit"]},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return path


def _safe_name(value: str) -> str:
    result = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in result.split("-") if part) or "model"


def _flatten(value, prefix=""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def refined_structure_from_candidate(candidate, dataset: DiffractionDataset) -> Structure:
    """Reconstruct a refined structure from the exact stored raw checkpoint."""
    policy = candidate.provenance["policy"]
    checkpoint = candidate.provenance["checkpoint"]["raw_groups"]
    calculator = BraggCalculator(
        mode=dataset.radiation,
        wavelength=dataset.wavelength,
        two_theta_range=(float(dataset.coordinate[0]), float(dataset.coordinate[-1])),
        primitive=False,
    ).load(candidate.structure)
    backend = calculator.backend
    parameters = calculator.tensor_parameters()
    raw = {name: np.asarray(value, dtype=np.float64) for name, value in checkpoint.items()}
    lattice_model = calculator.symmetry_lattice_parameterization()
    if "lattice" in raw:
        parameters["lattice"] = lattice_model.expand(raw["lattice"], backend)
    coordinate_model = calculator.symmetry_coordinate_parameterization()
    if "coordinates" in raw:
        parameters["frac_coords"] = coordinate_model.expand(raw["coordinates"], backend)
    if "rigid_bodies" in raw:
        rigid = calculator.rigid_body_parameterization(
            policy["rigid_bodies"],
            translation_scale=policy["rigid_translation_scale"],
            rotation_scale_degrees=policy["rigid_rotation_scale_degrees"],
        )
        parameters["frac_coords"] = rigid.expand(
            raw["rigid_bodies"], backend, lattice=parameters["lattice"]
        )
    if "occupancies" in raw:
        occupancy = calculator.symmetry_occupancy_parameterization(
            mode=policy["occupancy_mode"]
        )
        parameters["occupancies"] = occupancy.expand(raw["occupancies"], backend)
    if "b_iso" in raw:
        model = calculator.symmetry_b_iso_parameterization(
            default_if_zero=policy["default_b_iso"]
        )
        parameters["b_iso"] = model.expand(raw["b_iso"], backend)
        parameters.pop("u_cart", None)
    if "u_aniso" in raw:
        model = calculator.symmetry_u_aniso_parameterization(
            default_u_iso=policy["default_u_iso"]
        )
        parameters["u_cart"] = model.expand(raw["u_aniso"], backend)
    arrays = {name: np.asarray(value) for name, value in parameters.items()}
    site_indices = calculator._symm["site_indices"]
    symbols = calculator._symm["symbols"]
    species, coordinates, b_values, u_values = [], [], [], []
    for site_index in range(len(calculator._symm["structure"])):
        contributions = np.flatnonzero(site_indices == site_index)
        species.append(
            {
                symbols[index]: float(arrays["occupancies"][index])
                for index in contributions
                if arrays["occupancies"][index] > 1e-10
            }
        )
        representative = int(contributions[0])
        coordinates.append(arrays["frac_coords"][representative] % 1.0)
        if "b_iso" in arrays:
            b_values.append(float(arrays["b_iso"][representative]))
        if "u_cart" in arrays:
            u_values.append(arrays["u_cart"][representative].tolist())
    properties = {}
    if b_values:
        properties["B_iso"] = b_values
    if u_values:
        properties["U_cart"] = u_values
    return Structure(
        arrays["lattice"], species, coordinates, site_properties=properties,
        coords_are_cartesian=False,
    )
