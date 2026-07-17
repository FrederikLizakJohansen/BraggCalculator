import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "complete_characterization_tutorial.ipynb"
SIMPLE_NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "simple_refinement_tutorial.ipynb"
PROGRESSIVE_NOTEBOOK = (
    Path(__file__).parents[1] / "notebooks" / "progressive_refinement_tutorial.ipynb"
)
ADVANCED_SESSION_NOTEBOOK = (
    Path(__file__).parents[1] / "notebooks" / "advanced_session_tutorial.ipynb"
)


def test_complete_tutorial_is_executed_and_covers_the_scientist_workflow():
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    cells = document["cells"]
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    for phrase in (
        "Forward simulation",
        "information-loss",
        "amplitude–phase mismatch",
        "Experimental discriminability",
        "staged refinement policy",
        "Identifiability",
        "Design a better measurement",
        "parallel guided UI",
    ):
        assert phrase.lower() in markdown.lower()
    for call in (
        "DiffractionDataset.from_xye",
        "BraggCalculator",
        "diagnose_structures",
        "compare_calculators",
        "ProjectStore.create",
        "store.run()",
        "store.run(resume=True)",
        "suggest_measurements",
    ):
        assert call in code
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert sum(
        output.get("output_type") in {"display_data", "execute_result"}
        for cell in code_cells
        for output in cell.get("outputs", [])
    ) >= 10


def test_simple_tutorial_is_executed_and_stays_focused_on_one_refinement():
    document = json.loads(SIMPLE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = document["cells"]
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    for call in (
        "Structure.from_file",
        "DiffractionDataset.from_xye",
        "RefinementPolicy.quick",
        "RefinementSession",
        "session.run(policy)",
    ):
        assert call in code
    assert "diagnose_structures" not in code
    assert "compare_calculators" not in code
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    ) == 2


def test_progressive_tutorial_executes_every_refinement_level_and_figure():
    document = json.loads(PROGRESSIVE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = document["cells"]
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    for phrase in (
        "scale and background",
        "peak profile, and lattice",
        "atomic coordinates",
        "joint polish",
        "loss curve",
        "Other refinement types",
    ):
        assert phrase.lower() in markdown.lower()
    for call in (
        "OptimizationStage",
        "RefinementPolicy",
        "RefinementSession",
        "session.run(policy)",
        "refined_structure_from_candidate",
    ):
        assert call in code
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    ) == 4


def test_advanced_session_tutorial_executes_real_data_session_api():
    document = json.loads(ADVANCED_SESSION_NOTEBOOK.read_text(encoding="utf-8"))
    cells = document["cells"]
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    )
    for phrase in (
        "NIST SRM 660c",
        "CandidateRefinementResult",
        "SessionResult",
        "checkpoint",
        "PhaseMixtureSession",
        "Structural parameter families",
    ):
        assert phrase.lower() in markdown.lower()
    for call in (
        "DiffractionDataset.from_xye",
        "dataset.select_range",
        "dataset.exclude",
        "RefinementPolicy.quick",
        "RefinementPolicy.cautious",
        "RefinementPolicy.robust",
        "RefinementSession",
        "session.run(profile_policy)",
        "session.refine_candidate",
        "refined_structure_from_candidate",
        "session_result_to_dict",
        "session.write_html",
        "PhaseMixtureSession",
    ):
        assert call in code
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    ) == 6
