import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "complete_characterization_tutorial.ipynb"


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
