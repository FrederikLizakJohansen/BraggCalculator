import json
from hashlib import sha256
from importlib.resources import files

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from braggcalculator import (
    BraggCalculator,
    DiagnosticService,
    DiffractionDataset,
    OptimizationStage,
    ProjectStore,
    RefinementPolicy,
    RefinementSession,
    policy_from_dict,
    policy_to_dict,
)
from braggcalculator.mcp import call_tool


def _interface_inputs(tmp_path):
    structure_a = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    structure_b = structure_a.copy()
    structure_b.translate_sites([1], [0.02, 0, 0], frac_coords=True)
    model_a = tmp_path / "a.cif"
    model_b = tmp_path / "b.cif"
    CifWriter(structure_a).write_file(model_a)
    CifWriter(structure_b).write_file(model_b)
    generator = BraggCalculator(
        primitive=False, two_theta_range=(20.0, 45.0), two_theta_step=0.2
    ).load(structure_a)
    coordinate, profile = generator.pattern()
    intensity = 0.0005 * profile + 3.0
    data = tmp_path / "pattern.xye"
    np.savetxt(data, np.column_stack([coordinate, intensity, np.sqrt(intensity)]))
    return data, model_a, model_b, structure_a


def _tiny_policy():
    return RefinementPolicy(
        refine_lattice=True,
        background_degree=0,
        diagnostic_points=0,
        holdout_stride=3,
        stages=(
            OptimizationStage(
                "tiny", ("scale", "background", "profile", "lattice"), 3, 0.002
            ),
        ),
    )


def test_policy_round_trip_preserves_stages():
    policy = _tiny_policy()
    restored = policy_from_dict(policy_to_dict(policy))
    assert restored == policy
    with pytest.raises(ValueError, match="unknown refinement policy"):
        policy_from_dict({"made_up": True})


def test_session_checkpoint_is_exactly_resumable(tmp_path):
    data, _, _, structure = _interface_inputs(tmp_path)
    dataset = DiffractionDataset.from_xye(data, wavelength=1.5406)
    session = RefinementSession(dataset, [structure], names=["candidate"])
    first = session.run(_tiny_policy()).candidates[0]
    checkpoint = first.provenance["checkpoint"]
    second = session.run(
        _tiny_policy(), checkpoints={"candidate": checkpoint}
    ).candidates[0]
    assert checkpoint["format"] == "braggcalculator.raw-parameter-state/v1"
    assert second.provenance["resumed_from_checkpoint"]
    expected_digest = sha256(
        json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert second.provenance["resume_checkpoint_sha256"] == expected_digest
    assert set(second.provenance["checkpoint"]["raw_groups"]) == set(
        checkpoint["raw_groups"]
    )
    invalid = json.loads(json.dumps(checkpoint))
    invalid["raw_groups"].pop("profile")
    with pytest.raises(ValueError, match="do not match policy"):
        session.run(_tiny_policy(), checkpoints={"candidate": invalid})


def test_project_run_resume_workspace_and_exports(tmp_path):
    data, model_a, model_b, _ = _interface_inputs(tmp_path)
    store = ProjectStore.create(
        tmp_path / "project",
        dataset_path=data,
        model_paths=[model_a, model_b],
        names=["reference", "shifted"],
        wavelength=1.5406,
        title="Linked interface test",
        policy=_tiny_policy(),
    )
    document, first = store.run()
    assert set(first.ranking) == {"reference", "shifted"}
    run = document["runs"][-1]
    for artifact in (
        "result_json", "profiles_csv", "parameters_csv", "refined_cif",
        "workspace_html", "audit_json",
    ):
        assert artifact in run["artifacts"]
    workspace = store.directory / run["artifacts"]["workspace_html"]
    text = workspace.read_text(encoding="utf-8")
    assert "Mismatch disk" in text
    assert "Parameters, constraints and release state" in text
    assert "braggcalculator.workspace/v1" in text
    for path in run["artifacts"]["refined_cif"].values():
        assert Structure.from_file(store.directory / path)
    refined = Structure.from_file(
        store.directory / run["artifacts"]["refined_cif"]["reference"]
    )
    physical_a = next(item for item in first.candidates if item.name == "reference")
    assert refined.lattice.a == pytest.approx(
        physical_a.physical_parameters["cell_parameters"]["a"], rel=1e-6
    )

    resumed, second = store.run(resume=True)
    latest = resumed["runs"][-1]
    assert latest["parent_run_id"] == "run-0001"
    assert latest["resumed"]
    assert all(item["resumed_from_checkpoint"] for item in latest["trace_segments"])
    assert second.candidates
    result = store.read_result()
    assert result["schema"] == "braggcalculator.session-result/v1"
    assert result["run_id"] == "run-0002"
    sensitivity = DiagnosticService(tmp_path).dispatch(
        "analyze_sensitivity", {"project": "project", "candidate": "reference"}
    )
    assert sensitivity["result"]["candidates"][0]["name"] == "reference"
    assert sensitivity["result"]["candidates"][0]["recommendation"]


def test_service_and_mcp_enforce_scoped_projects_and_release_acknowledgement(tmp_path):
    data, model_a, _, _ = _interface_inputs(tmp_path)
    service = DiagnosticService(tmp_path / "service")
    simulated = service.dispatch("simulate_pattern", {"structure_path": str(model_a)})
    assert simulated["schema"] == "braggcalculator.service-response/v1"
    assert simulated["result"]["reflections"]["hkl"]
    with pytest.raises(ValueError, match="escapes"):
        service.dispatch("project_status", {"project": "../outside"})

    arguments = {
        "project": "agent-project",
        "dataset_path": str(data),
        "model_paths": [str(model_a)],
        "wavelength_angstrom": 1.5406,
        "policy": policy_to_dict(
            RefinementPolicy.quick(refine_coordinates=True)
        ),
        "release_policy_acknowledged": False,
    }
    with pytest.raises(ValueError, match="may not silently release"):
        call_tool(service, "bragg_create_project", arguments)
    arguments["release_policy_acknowledged"] = True
    created = call_tool(service, "bragg_create_project", arguments)
    assert created["result"]["project"] == "agent-project"


def test_project_detects_mutated_input(tmp_path):
    data, model_a, _, _ = _interface_inputs(tmp_path)
    store = ProjectStore.create(
        tmp_path / "project", dataset_path=data, model_paths=[model_a],
        wavelength=1.5406, policy=_tiny_policy(),
    )
    copied = store.directory / store.load()["dataset"]["path"]
    copied.write_text(copied.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum changed"):
        store.run()


def test_versioned_json_schemas_are_packaged():
    package = files("braggcalculator") / "schemas" / "v1"
    project = json.loads((package / "project.schema.json").read_text())
    result = json.loads((package / "result.schema.json").read_text())
    audit = json.loads((package / "audit.schema.json").read_text())
    service = json.loads((package / "service-response.schema.json").read_text())
    assert project["properties"]["schema"]["const"] == "braggcalculator.project/v1"
    assert result["properties"]["schema"]["const"] == "braggcalculator.session-result/v1"
    assert audit["properties"]["schema"]["const"] == "braggcalculator.audit/v1"
    assert service["properties"]["schema"]["const"] == "braggcalculator.service-response/v1"
