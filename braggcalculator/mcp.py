"""MCP-compatible tool declarations and a minimal JSON-RPC stdio server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .service import DiagnosticService


MCP_PROTOCOL_VERSION = "2025-06-18"


def _object_schema(required, properties):
    return {
        "type": "object", "required": required, "properties": properties,
        "additionalProperties": False,
    }


MCP_TOOLS = (
    {
        "name": "bragg_create_project",
        "description": "Create a checksummed refinement project with an explicit release policy.",
        "inputSchema": _object_schema(
            ["project", "dataset_path", "model_paths", "wavelength_angstrom", "policy",
             "release_policy_acknowledged"],
            {
                "project": {"type": "string"}, "dataset_path": {"type": "string"},
                "model_paths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "wavelength_angstrom": {"type": "number", "exclusiveMinimum": 0},
                "policy": {"type": "object"},
                "release_policy_acknowledged": {"type": "boolean"},
                "names": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"}, "radiation": {"enum": ["xray", "neutron"]},
            },
        ),
    },
    {
        "name": "bragg_run_project",
        "description": "Run a saved project and return ranked evidence and artifact paths.",
        "inputSchema": _object_schema(["project"], {"project": {"type": "string"}}),
    },
    {
        "name": "bragg_resume_project",
        "description": "Continue from the exact raw checkpoint of the latest project run.",
        "inputSchema": _object_schema(["project"], {"project": {"type": "string"}}),
    },
    {
        "name": "bragg_project_status",
        "description": "Inspect project policy, run lineage, traces, and audit state.",
        "inputSchema": _object_schema(["project"], {"project": {"type": "string"}}),
    },
    {
        "name": "bragg_project_result",
        "description": "Read a versioned full result for the latest or selected project run.",
        "inputSchema": _object_schema(
            ["project"],
            {"project": {"type": "string"}, "run_id": {"type": "string"}},
        ),
    },
    {
        "name": "bragg_analyze_sensitivity",
        "description": "Return identifiability, informative regions, warnings, and next-step guidance.",
        "inputSchema": _object_schema(
            ["project"],
            {
                "project": {"type": "string"}, "run_id": {"type": "string"},
                "candidate": {"type": "string"},
            },
        ),
    },
    {
        "name": "bragg_simulate_pattern",
        "description": "Simulate a line table and powder profile for one CIF model.",
        "inputSchema": _object_schema(
            ["structure_path"],
            {
                "structure_path": {"type": "string"},
                "radiation": {"enum": ["xray", "neutron"]},
                "wavelength_angstrom": {"type": "number"},
                "domain": {"enum": ["two_theta", "q"]},
            },
        ),
    },
    {
        "name": "bragg_compare_models",
        "description": "Run relationship-aware structural and diffraction diagnostics.",
        "inputSchema": _object_schema(
            ["structure_a", "structure_b"],
            {
                "structure_a": {"type": "string"}, "structure_b": {"type": "string"},
                "radiation": {"enum": ["xray", "neutron"]},
                "wavelength_angstrom": {"type": "number"},
                "profile_fwhm_q": {"type": "number"},
            },
        ),
    },
    {
        "name": "bragg_suggest_measurement",
        "description": "Rank declared experiments for discriminating two candidate structures.",
        "inputSchema": _object_schema(
            ["structure_a", "structure_b", "experiments"],
            {
                "structure_a": {"type": "string"}, "structure_b": {"type": "string"},
                "experiments": {"type": "array", "items": {"type": "object"}, "minItems": 1},
            },
        ),
    },
)


_TO_OPERATION = {
    "bragg_create_project": "create_project",
    "bragg_run_project": "run_project",
    "bragg_resume_project": "resume_project",
    "bragg_project_status": "project_status",
    "bragg_project_result": "project_result",
    "bragg_analyze_sensitivity": "analyze_sensitivity",
    "bragg_simulate_pattern": "simulate_pattern",
    "bragg_compare_models": "compare_models",
    "bragg_suggest_measurement": "suggest_measurement",
}


def call_tool(service: DiagnosticService, name: str, arguments: dict):
    if name not in _TO_OPERATION:
        raise KeyError(f"unknown MCP tool: {name}")
    payload = dict(arguments)
    if name == "bragg_create_project":
        acknowledged = payload.pop("release_policy_acknowledged", False)
        policy = payload.get("policy", {})
        structural = sum(
            bool(policy.get(field))
            for field in ("refine_coordinates", "refine_b_iso", "refine_u_aniso", "rigid_bodies")
        ) + (policy.get("occupancy_mode", "fixed") != "fixed")
        if structural and not acknowledged:
            raise ValueError(
                "structural release requires release_policy_acknowledged=true; an agent may not "
                "silently release structural parameters"
            )
    return service.dispatch(_TO_OPERATION[name], payload)


def serve_stdio(root):
    service = DiagnosticService(root)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            method = request.get("method")
            if method == "initialize":
                result = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "braggcalculator", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": list(MCP_TOOLS)}
            elif method == "tools/call":
                params = request.get("params", {})
                value = call_tool(service, params["name"], params.get("arguments", {}))
                result = {
                    "content": [{"type": "text", "text": json.dumps(value)}],
                    "structuredContent": value,
                    "isError": False,
                }
            elif method == "notifications/initialized":
                continue
            else:
                raise KeyError(f"unsupported JSON-RPC method: {method}")
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        except Exception as error:
            response = {
                "jsonrpc": "2.0", "id": request.get("id") if "request" in locals() else None,
                "error": {"code": -32602, "message": f"{type(error).__name__}: {error}"},
            }
        print(json.dumps(response), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("bragg-projects"))
    args = parser.parse_args(argv)
    serve_stdio(args.root)


if __name__ == "__main__":
    main()
