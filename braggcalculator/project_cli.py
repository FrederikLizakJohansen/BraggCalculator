"""Command-line lifecycle for portable BraggCalculator projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .project import ProjectStore, policy_from_dict


def build_parser():
    parser = argparse.ArgumentParser(prog="bragg-project", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="copy and checksum inputs into a project")
    create.add_argument("project", type=Path)
    create.add_argument("--data", type=Path, required=True)
    create.add_argument("--model", type=Path, action="append", required=True)
    create.add_argument("--name", action="append")
    create.add_argument("--wavelength", type=float, required=True)
    create.add_argument("--radiation", choices=("xray", "neutron"), default="xray")
    create.add_argument("--policy", type=Path, help="versioned policy JSON object")
    create.add_argument("--title", default="BraggCalculator project")
    for name in ("run", "resume", "status"):
        command = commands.add_parser(name)
        command.add_argument("project", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "create":
        policy = None
        if args.policy is not None:
            policy = policy_from_dict(json.loads(args.policy.read_text(encoding="utf-8")))
        store = ProjectStore.create(
            args.project,
            dataset_path=args.data,
            model_paths=args.model,
            names=args.name,
            wavelength=args.wavelength,
            radiation=args.radiation,
            title=args.title,
            policy=policy,
        )
        document = store.load()
        print(f"created {store.project_path}")
        print(f"project id: {document['project_id']}")
        return 0
    store = ProjectStore(args.project)
    if args.command in {"run", "resume"}:
        document, result = store.run(resume=args.command == "resume")
        run = document["runs"][-1]
        print(f"completed {run['run_id']}: {result.conclusion}")
        print(f"workspace: {store.directory / run['artifacts']['workspace_html']}")
        return 0
    document = store.load()
    print(json.dumps({"project_id": document["project_id"], "runs": document["runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
