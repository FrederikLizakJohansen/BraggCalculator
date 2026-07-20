#!/usr/bin/env python3
"""Build the frozen, crystal-system-balanced COD validation corpus.

This is a release-maintainer script, not part of routine validation. It queries
COD search-result pages, chooses candidates using a fixed hash seed, downloads
them in one rsync transaction, and writes immutable CIFs plus their manifest.
Agreement with BraggCalculator is deliberately not evaluated during selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
COD_RESULTS = "https://www.crystallography.net/cod/result.php"
COD_RSYNC = "rsync://www.crystallography.net/cif/"
DEFAULT_SEED = "braggcalculator-cod-corpus-v1"

# Ten predetermined strata per declared crystal system. Repeated triclinic
# groups are necessary because that system has only two space groups.
TARGETS = {
    "triclinic": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
    "monoclinic": (3, 4, 5, 6, 7, 8, 9, 11, 13, 15),
    "orthorhombic": (16, 22, 29, 35, 42, 48, 55, 61, 68, 74),
    "tetragonal": (75, 82, 90, 97, 105, 112, 120, 127, 135, 142),
    "trigonal": (143, 146, 149, 152, 155, 158, 161, 164, 166, 167),
    "hexagonal": (168, 171, 174, 177, 180, 183, 186, 189, 192, 194),
    "cubic": (195, 199, 203, 207, 211, 215, 219, 223, 227, 230),
}


def _fetch(url: str, data: bytes | None = None) -> str:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "BraggCalculator publication validation/0.1"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def _target_tasks() -> list[tuple[str, int, int]]:
    tasks = []
    for crystal_system, groups in TARGETS.items():
        occurrences: dict[int, int] = {}
        for space_group_number in groups:
            ordinal = occurrences.get(space_group_number, 0)
            occurrences[space_group_number] = ordinal + 1
            tasks.append((crystal_system, ordinal, space_group_number))
    return tasks


def _candidate_ids(
    task: tuple[str, int, int], seed: str, candidate_count: int
) -> tuple[tuple[str, int, int], list[str]]:
    crystal_system, ordinal, space_group_number = task
    body = urllib.parse.urlencode(
        {"space_group_number": str(space_group_number), "submit": "Send"}
    ).encode()
    initial = _fetch(COD_RESULTS, body)
    session_match = re.search(r"CODSESSION=([a-zA-Z0-9]+)", initial)
    if not session_match:
        return task, []
    pages = [int(value) for value in re.findall(r"(?:[?&]|&amp;)page=(\d+)", initial)] or [0]
    token = hashlib.sha256(
        f"{seed}:{crystal_system}:{space_group_number}:{ordinal}".encode()
    ).digest()
    page = int.from_bytes(token[:8], "big") % (max(pages) + 1)
    query = urllib.parse.urlencode(
        {
            "CODSESSION": session_match.group(1),
            "count": 20,
            "page": page,
            "order_by": "file",
            "order": "asc",
        }
    )
    result = _fetch(f"{COD_RESULTS}?{query}")
    identifiers = list(dict.fromkeys(re.findall(r'href="(\d{7})\.cif"', result)))
    identifiers.sort(
        key=lambda identifier: hashlib.sha256(
            f"{seed}:{crystal_system}:{space_group_number}:{ordinal}:{identifier}".encode()
        ).digest()
    )
    return task, identifiers[:candidate_count]


def _remote_path(identifier: str) -> str:
    return f"{identifier[0]}/{identifier[1:3]}/{identifier[3:5]}/{identifier}.cif"


def _revision(cif_path: Path) -> int | None:
    match = re.search(
        r"^#\$Revision:\s*(\d+)\s*\$", cif_path.read_text(errors="replace"), re.MULTILINE
    )
    return int(match.group(1)) if match else None


def build_corpus(
    output_dir: Path,
    *,
    seed: str,
    accessed_date: str,
    max_sites: int,
    candidate_count: int,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty corpus directory: {output_dir}")
    tasks = _target_tasks()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        gathered = dict(
            pool.map(
                lambda task: _candidate_ids(task, seed, candidate_count),
                tasks,
            )
        )
    all_identifiers = sorted(
        {identifier for identifiers in gathered.values() for identifier in identifiers}
    )

    temporary = Path(tempfile.mkdtemp(prefix="bragg-cod-corpus-"))
    try:
        file_list = temporary / "files.txt"
        file_list.write_text(
            "\n".join(_remote_path(identifier) for identifier in all_identifiers) + "\n"
        )
        subprocess.run(
            [
                "rsync",
                "-a",
                "--files-from",
                str(file_list),
                COD_RSYNC,
                f"{temporary / 'download'}/",
            ],
            check=True,
        )
        downloaded = {path.stem: path for path in (temporary / "download").rglob("*.cif")}
        selections = []
        for task in tasks:
            crystal_system, _, space_group_number = task
            selected = None
            rejected = []
            for identifier in gathered[task]:
                path = downloaded.get(identifier)
                if path is None:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        structure = Structure.from_file(path)
                except Exception as error:
                    rejected.append((identifier, type(error).__name__))
                    continue
                symbols = {element.symbol for element in structure.composition.elements}
                if symbols - set(ATOMIC_SCATTERING_PARAMS) or symbols - set(
                    ATOMIC_SCATTERING_LEN
                ):
                    rejected.append((identifier, "unsupported element"))
                    continue
                if not 1 <= len(structure) <= max_sites:
                    rejected.append((identifier, f"{len(structure)} sites"))
                    continue
                selected = (identifier, path, structure)
                break
            if selected is None:
                raise RuntimeError(f"no eligible CIF for {task}: {rejected}")
            selections.append((crystal_system, space_group_number, *selected))

        identifiers = [selection[2] for selection in selections]
        if len(set(identifiers)) != len(identifiers):
            duplicates = sorted(
                {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
            )
            raise RuntimeError(f"selection produced duplicate COD identifiers: {duplicates}")

        cif_dir = output_dir / "cifs"
        cif_dir.mkdir(parents=True, exist_ok=True)
        cases = []
        for crystal_system, space_group_number, identifier, source, structure in selections:
            destination = cif_dir / f"{identifier}.cif"
            shutil.copyfile(source, destination)
            revision = _revision(destination)
            source_url = f"https://www.crystallography.net/cod/{identifier}.cif"
            if revision is not None:
                source_url += f"@{revision}"
            cases.append(
                {
                    "id": f"COD-{identifier}",
                    "cod_id": identifier,
                    "revision": revision,
                    "path": f"cifs/{identifier}.cif",
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "source_url": source_url,
                    "crystal_system": crystal_system,
                    "space_group_number": space_group_number,
                    "input_sites": len(structure),
                    "disordered": not structure.is_ordered,
                }
            )
    finally:
        shutil.rmtree(temporary)

    manifest = {
        "schema_version": 1,
        "name": "BraggCalculator COD powder-pattern validation corpus 1.0.0",
        "source": {
            "name": "Crystallography Open Database",
            "homepage": "https://www.crystallography.net/cod/",
            "license": "CC0-1.0",
            "accessed_utc": accessed_date,
        },
        "selection": {
            "seed": seed,
            "description": (
                "Ten records per declared crystal system, distributed across space-group "
                "numbers. Within each target stratum a deterministic hash selects a result "
                "page and candidate order."
            ),
            "eligibility": (
                "Non-retracted, non-error, non-theoretical, non-duplicate COD records "
                f"parseable by pymatgen, with 1-{max_sites} supplied sites and elements "
                "tabulated by both pymatgen XRDCalculator and NDCalculator. Agreement with "
                "BraggCalculator was not an eligibility criterion."
            ),
        },
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "cif_validation")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--accessed-date", required=True)
    parser.add_argument("--max-sites", type=int, default=1024)
    parser.add_argument("--candidate-count", type=int, default=12)
    args = parser.parse_args()
    if args.max_sites <= 0 or not 1 <= args.candidate_count <= 20:
        parser.error("--max-sites must be positive and --candidate-count must be 1-20")
    manifest = build_corpus(
        args.output_dir,
        seed=args.seed,
        accessed_date=args.accessed_date,
        max_sites=args.max_sites,
        candidate_count=args.candidate_count,
    )
    print(f"wrote {args.output_dir / 'manifest.json'} with {len(manifest['cases'])} cases")


if __name__ == "__main__":
    main()
