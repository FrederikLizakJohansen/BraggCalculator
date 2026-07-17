"""Guided local web application for complete diffraction-analysis projects."""

from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from urllib.parse import unquote, urlparse
from uuid import uuid4

import numpy as np
from pymatgen.core import Structure

from .core import BraggCalculator
from .diagnostics import compare_calculators
from .optimization import OptimizationStage
from .project import ProjectStore
from .session import RefinementPolicy
from .structural_diagnostics import diagnose_structures, suggest_measurements


UI_SCHEMA = "braggcalculator.guided-ui/v1"
_PROJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _validate_project_name(value: str) -> str:
    value = str(value)
    if not _PROJECT_NAME.fullmatch(value):
        raise ValueError(
            "project must contain 1-64 letters, digits, underscores or hyphens and "
            "must start with a letter or digit"
        )
    return value


def _safe_upload_name(value: str, suffixes: set[str]) -> str:
    name = Path(str(value)).name
    if not name or Path(name).suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ValueError(f"uploaded file must use one of these suffixes: {expected}")
    return name


def _tutorial_policy() -> RefinementPolicy:
    return RefinementPolicy(
        background_degree=1,
        refine_lattice=True,
        refine_coordinates=True,
        coordinate_restraint=0.05,
        holdout_stride=8,
        diagnostic_points=32,
        stages=(
            OptimizationStage("scale/background", ("scale", "background"), 25, 0.025),
            OptimizationStage("profile/lattice", ("profile", "lattice"), 35, 0.008),
            OptimizationStage("coordinates", ("coordinates",), 35, 0.004),
            OptimizationStage(
                "joint",
                ("scale", "background", "profile", "lattice", "coordinates"),
                45,
                0.002,
            ),
        ),
    )


def _policy_from_ui(payload: Mapping[str, Any]) -> RefinementPolicy:
    recipe = str(payload.get("recipe", "quick"))
    refine_coordinates = bool(payload.get("refine_coordinates", False))
    occupancy_mode = str(payload.get("occupancy_mode", "fixed"))
    refine_b_iso = bool(payload.get("refine_b_iso", False))
    refine_u_aniso = bool(payload.get("refine_u_aniso", False))
    if recipe == "tutorial":
        policy = _tutorial_policy()
    elif recipe == "cautious":
        policy = RefinementPolicy.cautious(
            refine_coordinates=refine_coordinates,
            occupancy_mode=occupancy_mode,
            refine_b_iso=refine_b_iso,
            refine_u_aniso=refine_u_aniso,
        )
    elif recipe == "quick":
        policy = RefinementPolicy.quick(
            refine_coordinates=refine_coordinates,
            occupancy_mode=occupancy_mode,
            refine_b_iso=refine_b_iso,
            refine_u_aniso=refine_u_aniso,
        )
    else:
        raise ValueError("recipe must be 'quick', 'cautious', or 'tutorial'")
    return replace(
        policy,
        background_degree=int(payload.get("background_degree", policy.background_degree)),
        refine_lattice=bool(payload.get("refine_lattice", policy.refine_lattice)),
        coordinate_restraint=float(
            payload.get("coordinate_restraint", policy.coordinate_restraint)
        ),
        diagnostic_points=int(payload.get("diagnostic_points", policy.diagnostic_points)),
    )


class GuidedUI:
    """Application service used by both the HTTP handler and direct tests."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _store(self, project: str) -> ProjectStore:
        identifier = _validate_project_name(project)
        return ProjectStore(self.root / identifier)

    def create_example(self, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {} if payload is None else dict(payload)
        project = _validate_project_name(payload.get("project", f"tutorial-{uuid4().hex[:8]}"))
        resource = files("braggcalculator") / "tutorial_data"
        store = ProjectStore.create(
            self._store(project).directory,
            dataset_path=resource / "pattern.xye",
            model_paths=[resource / "model-a.cif", resource / "model-b.cif"],
            names=["reference motif", "oxygen-shift candidate"],
            wavelength=1.5406,
            title="Guided NaSiO2 candidate-refinement tutorial",
            radiation="xray",
            third_column="sigma",
            policy=_tutorial_policy(),
            metadata={
                "tutorial": True,
                "synthetic": True,
                "notice": (
                    "Synthetic teaching data with known compatible candidates; this is not "
                    "independent experimental validation."
                ),
            },
        )
        return self.project_summary(project, store=store)

    def create_uploaded_project(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        project = _validate_project_name(payload["project"])
        dataset = dict(payload["dataset"])
        models = [dict(item) for item in payload["models"]]
        if not models:
            raise ValueError("at least one CIF model must be uploaded")
        policy_options = dict(payload.get("policy", {}))
        policy = _policy_from_ui(policy_options)
        structural_release = (
            policy.refine_coordinates
            or policy.occupancy_mode != "fixed"
            or policy.refine_b_iso
            or policy.refine_u_aniso
        )
        if structural_release and not bool(payload.get("release_policy_acknowledged")):
            raise ValueError(
                "structural parameter release requires explicit acknowledgement in the UI"
            )
        staging = self.root / ".uploads" / uuid4().hex
        staging.mkdir(parents=True)
        try:
            dataset_name = _safe_upload_name(dataset["name"], {".xy", ".xye", ".dat", ".txt"})
            dataset_path = staging / dataset_name
            dataset_path.write_text(str(dataset["content"]), encoding="utf-8")
            model_paths, names = [], []
            for index, model in enumerate(models, start=1):
                name = _safe_upload_name(model["name"], {".cif"})
                path = staging / f"{index:02d}-{name}"
                path.write_text(str(model["content"]), encoding="utf-8")
                model_paths.append(path)
                names.append(str(model.get("label") or Path(name).stem))
            store = ProjectStore.create(
                self._store(project).directory,
                dataset_path=dataset_path,
                model_paths=model_paths,
                names=names,
                wavelength=float(payload.get("wavelength_angstrom", 1.5406)),
                title=str(payload.get("title") or project),
                radiation=str(payload.get("radiation", "xray")),
                third_column=str(payload.get("third_column", "sigma")),
                policy=policy,
                metadata={"tutorial": False, "uploaded_through": UI_SCHEMA},
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.project_summary(project, store=store)

    def project_summary(self, project: str, *, store: ProjectStore | None = None):
        store = self._store(project) if store is None else store
        document = store.load()
        return {
            "schema": UI_SCHEMA,
            "project": project,
            "project_id": document["project_id"],
            "title": document["title"],
            "models": [item["name"] for item in document["models"]],
            "dataset": document["dataset"],
            "policy": document["policy"],
            "runs": document["runs"],
            "audit": document["audit"],
        }

    def run(self, project: str, *, resume: bool = False) -> dict[str, Any]:
        document, result = self._store(project).run(resume=resume)
        record = document["runs"][-1]
        return {
            "schema": UI_SCHEMA,
            "project": project,
            "run_id": record["run_id"],
            "parent_run_id": record["parent_run_id"],
            "resumed": record["resumed"],
            "ranking": list(result.ranking),
            "conclusion": result.conclusion,
            "candidates": [
                {"name": item.name, "r_wp": item.r_wp, "chi_squared": item.chi_squared}
                for item in result.candidates
            ],
        }

    def diagnostics(self, project: str, run_id: str | None = None) -> dict[str, Any]:
        store = self._store(project)
        document = store.load()
        result = store.read_result(run_id)
        run = next(item for item in document["runs"] if item["run_id"] == result["run_id"])
        indices = np.linspace(
            0, len(result["dataset"]["coordinate"]) - 1,
            min(len(result["dataset"]["coordinate"]), 2600),
        ).astype(int)
        candidates = []
        for candidate in result["candidates"]:
            candidates.append(
                {
                    "name": candidate["name"],
                    "formula": candidate["formula"],
                    "r_wp": candidate["r_wp"],
                    "chi_squared": candidate["chi_squared"],
                    "held_out_r_wp": candidate["held_out_r_wp"],
                    "calculated": np.asarray(candidate["calculated"])[indices].tolist(),
                    "residual": np.asarray(candidate["residual"])[indices].tolist(),
                    "loss_history": candidate["loss_history"],
                    "stage_history": candidate["stage_history"],
                    "informative_regions": candidate["informative_regions"],
                    "identifiability": candidate["identifiability"],
                    "physical_parameters": candidate["physical_parameters"],
                    "recommendation": candidate["recommendation"],
                    "warnings": candidate["warnings"],
                    "convergence": candidate["convergence"],
                    "provenance": {
                        "dataset_sha256": candidate["provenance"]["dataset_sha256"],
                        "observation_uncertainty": candidate["provenance"][
                            "observation_uncertainty"
                        ],
                        "released_parameter_groups": candidate["provenance"]["policy"][
                            "released_parameter_groups"
                        ],
                        "resumed_from_checkpoint": candidate["provenance"][
                            "resumed_from_checkpoint"
                        ],
                    },
                }
            )
        model_paths = [(store.directory / item["path"]).resolve() for item in document["models"]]
        structures = [_structure_state(path, item["name"]) for path, item in zip(model_paths, document["models"])]
        comparison = _comparison_state(
            model_paths,
            radiation=document["dataset"]["radiation"],
            wavelength=float(document["dataset"]["wavelength_angstrom"]),
            two_theta=(
                float(result["dataset"]["coordinate"][0]),
                float(result["dataset"]["coordinate"][-1]),
            ),
        )
        measurements = _measurement_state(model_paths)
        artifact_urls = _artifact_urls(project, run["artifacts"])
        pairwise_items = list(result["pairwise_discrimination"].items())
        refined_discrimination = None
        if len(candidates) >= 2 and pairwise_items:
            uncertainty_model = candidates[0]["provenance"]["observation_uncertainty"][
                "model"
            ]
            pointwise = None
            if uncertainty_model == "independent marginal sigma":
                pointwise = (
                    (
                        np.asarray(candidates[0]["calculated"])
                        - np.asarray(candidates[1]["calculated"])
                    )
                    / np.asarray(result["dataset"]["sigma"])[indices]
                ) ** 2
            refined_discrimination = {
                "pair": pairwise_items[0][0],
                "total_delta_chi_squared": pairwise_items[0][1],
                "coordinate": np.asarray(result["dataset"]["coordinate"])[indices],
                "pointwise": pointwise,
                "uncertainty_model": uncertainty_model,
            }
        return _jsonable(
            {
                "schema": UI_SCHEMA,
                "project": project,
                "project_id": document["project_id"],
                "title": document["title"],
                "run_id": result["run_id"],
                "runs": document["runs"],
                "audit": document["audit"],
                "policy": document["policy"],
                "dataset": {
                    **result["dataset"],
                    "coordinate": np.asarray(result["dataset"]["coordinate"])[indices].tolist(),
                    "intensity": np.asarray(result["dataset"]["intensity"])[indices].tolist(),
                    "sigma": np.asarray(result["dataset"]["sigma"])[indices].tolist(),
                },
                "ranking": result["ranking"],
                "pairwise_discrimination": result["pairwise_discrimination"],
                "refined_discrimination": refined_discrimination,
                "conclusion": result["conclusion"],
                "candidates": candidates,
                "structures": structures,
                "comparison": comparison,
                "measurements": measurements,
                "artifacts": artifact_urls,
            }
        )

    def artifact(self, project: str, relative: str) -> Path:
        store = self._store(project)
        path = (store.directory / unquote(relative)).resolve()
        try:
            path.relative_to(store.directory)
        except ValueError as error:
            raise ValueError("artifact path escapes the project") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path


def _structure_state(path: Path, name: str) -> dict[str, Any]:
    structure = Structure.from_file(path)
    return {
        "name": name,
        "formula": structure.composition.reduced_formula,
        "lattice": {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
        },
        "sites": [
            {
                "index": index,
                "species": site.species_string,
                "fractional": np.asarray(site.frac_coords).tolist(),
            }
            for index, site in enumerate(structure)
        ],
    }


def _comparison_state(
    paths: list[Path], *, radiation: str, wavelength: float, two_theta: tuple[float, float]
) -> dict[str, Any]:
    if len(paths) < 2:
        return {"available": False, "reason": "Upload two or more candidates to compare them."}
    diagnostic = diagnose_structures(
        paths[0], paths[1], radiation=radiation, wavelength=wavelength,
        q_range=(0.5, 5.0), q_step=0.025, profile_fwhm_q=0.08,
        count_scale=100.0, background_density=1.0,
    )
    mismatch = None
    if diagnostic.mismatch is not None:
        calculator_a = BraggCalculator(
            mode=radiation, wavelength=wavelength, two_theta_range=two_theta, primitive=False
        ).load(paths[0])
        calculator_b = BraggCalculator(
            mode=radiation, wavelength=wavelength, two_theta_range=two_theta, primitive=False
        ).load(paths[1])
        exact = compare_calculators(calculator_a, calculator_b, domain="two_theta")
        table = calculator_a.reflection_table(domain="two_theta")
        source_indices = exact.match.indices_a
        keep = np.linspace(0, len(source_indices) - 1, min(len(source_indices), 1200)).astype(int)
        points = []
        for offset in keep:
            source = int(source_indices[offset])
            points.append(
                {
                    "hkl": exact.match.hkl[offset].tolist(),
                    "x": exact.x[offset], "y": exact.y[offset],
                    "radius": exact.radius[offset], "weight": exact.weights[offset],
                    "q": np.asarray(table.q)[source],
                    "two_theta": np.asarray(table.two_theta)[source],
                }
            )
        mismatch = {
            "d_sf": exact.d_sf,
            "d_amplitude": exact.d_amplitude,
            "d_phase": exact.d_phase,
            "origin_shift": exact.alignment.shift,
            "points": points,
        }
    peak_groups = []
    for label, groups in (("A", diagnostic.peak_groups_a), ("B", diagnostic.peak_groups_b)):
        peak_groups.append(
            {
                "candidate": label,
                "groups": [
                    {
                        "q_center": group.q_center,
                        "q_min": group.q_min,
                        "q_max": group.q_max,
                        "integrated_intensity": group.integrated_intensity,
                        "effective_reflections": group.effective_reflections,
                        "hkl": group.hkl,
                        "reflection_intensity": group.reflection_intensity,
                        "site_effects": group.site_effects,
                    }
                    for group in groups
                ],
            }
        )
    return {
        "available": True,
        "relationship": _jsonable(diagnostic.relationship.__dict__),
        "similarities": diagnostic.similarities,
        "dominant_information_loss": diagnostic.dominant_information_loss,
        "explanation": diagnostic.explanation,
        "declared_count_model_discrimination": (
            diagnostic.profile_discrimination.total_discrimination
        ),
        "discrimination_coordinate": diagnostic.profile_discrimination.coordinate,
        "pointwise_discrimination": diagnostic.profile_discrimination.pointwise_discrimination,
        "pair_distribution": {
            "radius": diagnostic.pair_distribution.radius,
            "a": diagnostic.pair_distribution.distribution_a,
            "b": diagnostic.pair_distribution.distribution_b,
            "similarity": diagnostic.pair_distribution.similarity,
        },
        "mismatch": mismatch,
        "peak_groups": peak_groups,
    }


def _measurement_state(paths: list[Path]) -> list[dict[str, Any]]:
    if len(paths) < 2:
        return []
    experiments = (
        {"name": "Laboratory Cu Kalpha", "radiation": "xray", "wavelength": 1.5406,
         "q_range": (0.5, 5.0), "q_step": 0.02, "fwhm_q": 0.12,
         "count_scale": 100.0, "background_density": 1.0},
        {"name": "High-resolution Cu Kalpha", "radiation": "xray", "wavelength": 1.5406,
         "q_range": (0.5, 5.0), "q_step": 0.01, "fwhm_q": 0.04,
         "count_scale": 100.0, "background_density": 1.0},
        {"name": "Neutron constant-wavelength", "radiation": "neutron", "wavelength": 1.8,
         "q_range": (0.5, 5.0), "q_step": 0.02, "fwhm_q": 0.08,
         "count_scale": 100.0, "background_density": 1.0},
        {"name": "Extended-Q high resolution", "radiation": "xray", "wavelength": 0.7,
         "q_range": (0.5, 8.0), "q_step": 0.01, "fwhm_q": 0.04,
         "count_scale": 100.0, "background_density": 1.0},
    )
    return [_jsonable(item.__dict__) for item in suggest_measurements(paths[0], paths[1], experiments)]


def _artifact_urls(project: str, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value):
        if isinstance(value, Mapping):
            return {key: convert(item) for key, item in value.items()}
        return f"/api/projects/{project}/artifacts/{value}"
    return {key: convert(value) for key, value in artifacts.items()}


def _content_type(path: Path) -> str:
    return {
        ".html": "text/html; charset=utf-8", ".json": "application/json",
        ".csv": "text/csv; charset=utf-8", ".cif": "text/plain; charset=utf-8",
    }.get(path.suffix.lower(), "application/octet-stream")


def _handler(application: GuidedUI):
    index = (files("braggcalculator") / "ui" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            content = json.dumps(_jsonable(value), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index)))
                self.end_headers()
                self.wfile.write(index)
                return
            if path == "/api/health":
                self._json(200, {"status": "ok", "schema": UI_SCHEMA})
                return
            match = re.fullmatch(r"/api/projects/([^/]+)/diagnostics", path)
            if match:
                return self._execute(lambda: application.diagnostics(match.group(1)))
            match = re.fullmatch(r"/api/projects/([^/]+)/artifacts/(.+)", path)
            if match:
                try:
                    artifact = application.artifact(match.group(1), match.group(2))
                    content = artifact.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", _content_type(artifact))
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header(
                        "Content-Disposition", f'attachment; filename="{artifact.name}"'
                    )
                    self.end_headers()
                    self.wfile.write(content)
                except (ValueError, FileNotFoundError) as error:
                    self._json(404, {"error": type(error).__name__, "message": str(error)})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > _MAX_UPLOAD_BYTES:
                    raise ValueError("request exceeds the 30 MiB local upload limit")
                payload = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/examples/tutorial":
                    return self._execute(lambda: application.create_example(payload))
                if path == "/api/projects":
                    return self._execute(lambda: application.create_uploaded_project(payload))
                match = re.fullmatch(r"/api/projects/([^/]+)/(run|resume)", path)
                if match:
                    return self._execute(
                        lambda: application.run(match.group(1), resume=match.group(2) == "resume")
                    )
                self._json(404, {"error": "not found"})
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})

        def _execute(self, callback):
            try:
                self._json(200, callback())
            except (KeyError, TypeError, ValueError, FileNotFoundError, FileExistsError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})

        def log_message(self, format, *args):
            return

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("bragg-ui-projects"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _handler(GuidedUI(args.root)))
    print(f"BraggCalculator guided UI listening on http://{args.host}:{args.port}")
    print("Trusted local use only: this server has no authentication or TLS.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBraggCalculator guided UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
