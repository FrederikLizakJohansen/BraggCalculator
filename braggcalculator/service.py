"""Versioned service operations and a dependency-free JSON HTTP transport."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import numpy as np

from .core import BraggCalculator
from .project import ProjectStore, policy_from_dict
from .structural_diagnostics import diagnose_structures, suggest_measurements


SERVICE_SCHEMA = "braggcalculator.service-response/v1"
SERVICE_OPERATIONS = (
    "create_project",
    "run_project",
    "resume_project",
    "project_status",
    "project_result",
    "analyze_sensitivity",
    "simulate_pattern",
    "compare_models",
    "suggest_measurement",
)


class DiagnosticService:
    """Stable high-level operations shared by HTTP, MCP, and local callers."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _project(self, identifier: str) -> ProjectStore:
        path = (self.root / identifier).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("project identifier escapes the service root") from error
        return ProjectStore(path)

    def dispatch(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if operation not in SERVICE_OPERATIONS:
            raise KeyError(f"unknown operation: {operation}")
        result = getattr(self, operation)(dict(payload))
        return {"schema": SERVICE_SCHEMA, "operation": operation, "result": result}

    def create_project(self, payload):
        policy = policy_from_dict(payload["policy"]) if "policy" in payload else None
        store = ProjectStore.create(
            self._project(payload["project"]).directory,
            dataset_path=payload["dataset_path"],
            model_paths=payload["model_paths"],
            wavelength=float(payload["wavelength_angstrom"]),
            names=payload.get("names"),
            title=payload.get("title", payload["project"]),
            radiation=payload.get("radiation", "xray"),
            data_format=payload.get("data_format", "xye"),
            third_column=payload.get("third_column", "sigma"),
            policy=policy,
            metadata=payload.get("metadata"),
        )
        document = store.load()
        return {
            "project": payload["project"],
            "project_id": document["project_id"],
            "schema": document["schema"],
            "model_names": [item["name"] for item in document["models"]],
        }

    def run_project(self, payload):
        document, result = self._project(payload["project"]).run(resume=False)
        return _run_summary(document, result)

    def resume_project(self, payload):
        document, result = self._project(payload["project"]).run(resume=True)
        return _run_summary(document, result)

    def project_status(self, payload):
        document = self._project(payload["project"]).load()
        return {
            "project_id": document["project_id"],
            "title": document["title"],
            "schema": document["schema"],
            "runs": document["runs"],
            "policy": document["policy"],
            "audit_events": len(document["audit"]),
        }

    def project_result(self, payload):
        return self._project(payload["project"]).read_result(payload.get("run_id"))

    def analyze_sensitivity(self, payload):
        result = self._project(payload["project"]).read_result(payload.get("run_id"))
        selected = payload.get("candidate")
        candidates = result["candidates"]
        if selected is not None:
            candidates = [item for item in candidates if item["name"] == selected]
            if not candidates:
                raise KeyError(f"unknown candidate: {selected}")
        return {
            "run_id": result["run_id"],
            "candidates": [
                {
                    "name": item["name"],
                    "identifiability": item["identifiability"],
                    "recommendation": item["recommendation"],
                    "warnings": item["warnings"],
                    "informative_regions": item["informative_regions"],
                }
                for item in candidates
            ],
        }

    def simulate_pattern(self, payload):
        domain = payload.get("domain", "two_theta")
        calculator = BraggCalculator(
            mode=payload.get("radiation", "xray"),
            wavelength=float(payload.get("wavelength_angstrom", 1.5406)),
            q_range=tuple(payload.get("q_range", (0.5, 8.0))),
            two_theta_range=tuple(payload.get("two_theta_range", (5.0, 120.0))),
            q_step=float(payload.get("step", 0.02)),
            two_theta_step=float(payload.get("step", 0.02)),
            primitive=bool(payload.get("primitive", False)),
        ).load(payload["structure_path"])
        coordinate, profile = calculator.pattern(domain=domain)
        table = calculator.reflection_table(domain=domain)
        return {
            "domain": domain,
            "radiation": calculator.mode,
            "wavelength_angstrom": calculator.wavelength,
            "coordinate": np.asarray(coordinate).tolist(),
            "profile": np.asarray(profile).tolist(),
            "reflections": {
                "hkl": table.hkl.tolist(),
                "q": np.asarray(table.q).tolist(),
                "two_theta": np.asarray(table.two_theta).tolist(),
                "intensity": np.asarray(table.intensity).tolist(),
            },
        }

    def compare_models(self, payload):
        result = diagnose_structures(
            payload["structure_a"], payload["structure_b"],
            radiation=payload.get("radiation", "xray"),
            wavelength=float(payload.get("wavelength_angstrom", 1.5406)),
            q_range=tuple(payload.get("q_range", (0.5, 5.0))),
            q_step=float(payload.get("q_step", 0.02)),
            profile_fwhm_q=float(payload.get("profile_fwhm_q", 0.08)),
            count_scale=float(payload.get("count_scale", 100.0)),
            background_density=float(payload.get("background_density", 1.0)),
        )
        mismatch = None
        if result.mismatch is not None:
            mismatch = {
                "d_sf": result.mismatch.d_sf,
                "d_amplitude": result.mismatch.d_amplitude,
                "d_phase": result.mismatch.d_phase,
                "matched_reflections": len(result.mismatch.match),
            }
        return {
            "relationship": {
                "regime": result.relationship.regime,
                "classification": result.relationship.classification,
                "complex_comparison_allowed": result.relationship.complex_comparison_allowed,
                "reason": result.relationship.reason,
            },
            "similarities": result.similarities,
            "dominant_information_loss": result.dominant_information_loss,
            "explanation": result.explanation,
            "total_discrimination": result.profile_discrimination.total_discrimination,
            "mismatch": mismatch,
            "largest_counterfactuals": [
                {
                    "name": item.name,
                    "effect_norm": item.effect_norm,
                    "alignment_fraction": item.alignment_fraction,
                }
                for item in sorted(
                    result.counterfactuals, key=lambda item: item.effect_norm, reverse=True
                )[:5]
            ],
        }

    def suggest_measurement(self, payload):
        recommendations = suggest_measurements(
            payload["structure_a"], payload["structure_b"], payload["experiments"]
        )
        return {
            "recommendations": [
                {
                    "name": item.name,
                    "radiation": item.radiation,
                    "wavelength_angstrom": item.wavelength,
                    "q_range": item.q_range,
                    "fwhm_q": item.fwhm_q,
                    "total_discrimination": item.total_discrimination,
                    "most_informative_q": item.most_informative_q,
                    "assumptions": item.assumptions,
                }
                for item in recommendations
            ]
        }


def _run_summary(document, result):
    run = document["runs"][-1]
    return {
        "run_id": run["run_id"],
        "parent_run_id": run["parent_run_id"],
        "resumed": run["resumed"],
        "ranking": list(result.ranking),
        "conclusion": result.conclusion,
        "candidates": [
            {
                "name": candidate.name,
                "r_wp": candidate.r_wp,
                "chi_squared": candidate.chi_squared,
                "recommendation": candidate.recommendation,
                "warnings": list(candidate.warnings),
            }
            for candidate in result.candidates
        ],
        "artifacts": run["artifacts"],
    }


def _handler(service: DiagnosticService):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, value):
            content = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._json(200, {"status": "ok", "schema": SERVICE_SCHEMA})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            prefix = "/v1/operations/"
            if not path.startswith(prefix):
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._json(200, service.dispatch(path[len(prefix) :], payload))
            except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
                self._json(400, {"error": type(error).__name__, "message": str(error)})

        def log_message(self, format, *args):
            return

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("bragg-projects"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _handler(DiagnosticService(args.root)))
    print(f"BraggCalculator service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
