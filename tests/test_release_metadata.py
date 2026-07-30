from pathlib import Path
import re
import tomllib

import braggcalculator


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    bibliography = (ROOT / "paper/paper.bib").read_text(encoding="utf-8")

    assert braggcalculator.__version__ == "0.4.0"
    assert project["tool"]["poetry"]["version"] == braggcalculator.__version__
    assert re.search(r"^version: 0\.4\.0$", citation, flags=re.MULTILINE)
    assert 'date-released: "2026-07-30"' in citation
    assert "version = {0.4.0}" in bibliography
