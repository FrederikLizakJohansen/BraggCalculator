import json
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import pytest

from braggcalculator.ui import GuidedUI, UI_SCHEMA, _handler


def _tutorial_payload(tmp_path, *, acknowledged):
    resources = Path(__file__).parents[1] / "braggcalculator" / "tutorial_data"
    return {
        "project": "uploaded-project",
        "title": "Uploaded tutorial inputs",
        "dataset": {
            "name": "pattern.xye",
            "content": (resources / "pattern.xye").read_text(),
        },
        "models": [
            {
                "name": "candidate.cif",
                "label": "candidate",
                "content": (resources / "model-a.cif").read_text(),
            }
        ],
        "wavelength_angstrom": 1.5406,
        "radiation": "xray",
        "third_column": "sigma",
        "policy": {"recipe": "quick", "refine_coordinates": True},
        "release_policy_acknowledged": acknowledged,
    }


def test_guided_ui_requires_acknowledgement_and_confines_artifacts(tmp_path):
    application = GuidedUI(tmp_path)
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        application.create_uploaded_project(_tutorial_payload(tmp_path, acknowledged=False))
    summary = application.create_uploaded_project(
        _tutorial_payload(tmp_path, acknowledged=True)
    )
    assert summary["project"] == "uploaded-project"
    assert summary["policy"]["refine_coordinates"]
    with pytest.raises(ValueError, match="escapes"):
        application.artifact("uploaded-project", "../../outside")
    with pytest.raises(ValueError, match="project must contain"):
        application.create_example({"project": "../outside"})


def test_tutorial_runs_and_populates_every_diagnostic_family(tmp_path):
    application = GuidedUI(tmp_path)
    summary = application.create_example({"project": "tutorial"})
    assert summary["dataset"]["metadata"]["synthetic"] is True
    completed = application.run("tutorial")
    assert completed["conclusion"] == (
        "The supplied experiment does not discriminate the refined candidates."
    )
    diagnostics = application.diagnostics("tutorial")
    assert diagnostics["schema"] == UI_SCHEMA
    assert len(diagnostics["candidates"]) == 2
    assert diagnostics["comparison"]["relationship"]["regime"] == "I"
    assert diagnostics["comparison"]["mismatch"]["points"]
    assert diagnostics["comparison"]["peak_groups"]
    assert diagnostics["comparison"]["pair_distribution"]["similarity"] > 0
    assert diagnostics["refined_discrimination"]["total_delta_chi_squared"] < 9
    assert diagnostics["refined_discrimination"]["pointwise"]
    assert diagnostics["comparison"]["declared_count_model_discrimination"] > 9
    assert diagnostics["candidates"][0]["identifiability"]["parameter_names"]
    assert len(diagnostics["measurements"]) == 4
    assert diagnostics["artifacts"]["refined_cif"]
    encoded = json.dumps(diagnostics, allow_nan=False)
    assert "Infinity" not in encoded


def test_local_http_serves_application_and_health(tmp_path):
    application = GuidedUI(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(application))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/", timeout=5) as response:
            html = response.read().decode()
        assert "Guided Characterization" in html
        assert html.count('class="explain"') >= 27
        assert "Amplitude &amp; phase" in html
        with urlopen(base + "/api/health", timeout=5) as response:
            health = json.load(response)
        assert health == {"status": "ok", "schema": UI_SCHEMA}
    finally:
        server.shutdown()
        server.server_close()
