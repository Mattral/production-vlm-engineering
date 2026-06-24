"""Unified CLI entry point: `production-vlm <command>` (also runnable as `python -m production_vlm.cli`).

Built on stdlib ``argparse`` so the CLI works in any environment with
just the package installed — no hard dependency on click/typer.
If ``rich`` is installed, output is upgraded with tables/coloring;
otherwise it falls back to plain ``print``.
"""

from __future__ import annotations

import argparse
import importlib
import sys

_EXAMPLES = {
    "vlm_chart_finetune": "examples.pipelines.vlm_chart_finetune.run",
    "vlm_edge_inference": "examples.pipelines.vlm_edge_inference.run",
    "embedding_drift_active_learning": "examples.pipelines.embedding_drift_active_learning.run",
    "vlm_robustness_guard": "examples.pipelines.vlm_robustness_guard.run",
    "vlm_video_temporal": "examples.pipelines.vlm_video_temporal.run",
}


def _print_table(rows: list[tuple[str, str]], title: str) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        table.add_column("Name", style="cyan")
        table.add_column("Module")
        for name, module in rows:
            table.add_row(name, module)
        Console().print(table)
        return
    except ImportError:
        pass
    print(f"\n{title}\n" + "-" * len(title))
    for name, module in rows:
        print(f"  {name:<36} {module}")


def cmd_list_examples(_: argparse.Namespace) -> int:
    _print_table(list(_EXAMPLES.items()), "Available Examples")
    return 0


def cmd_run_example(args: argparse.Namespace) -> int:
    if args.name not in _EXAMPLES:
        print(f"Unknown example '{args.name}'. Run `production-vlm list-examples`.", file=sys.stderr)
        return 1
    module = importlib.import_module(_EXAMPLES[args.name])
    module.main(config_path=args.config)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    if args.name not in _EXAMPLES:
        print(f"Unknown example '{args.name}'.", file=sys.stderr)
        return 1
    module = importlib.import_module(_EXAMPLES[args.name])
    if not hasattr(module, "benchmark"):
        print(f"{args.name} has no benchmark() entry point.", file=sys.stderr)
        return 1
    module.benchmark()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="production-vlm", description="Production VLM Engineering CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-examples", help="List available runnable examples").set_defaults(func=cmd_list_examples)

    run_p = sub.add_parser("run-example", help="Run a named example end-to-end")
    run_p.add_argument("name", help="Example name, see list-examples")
    run_p.add_argument("--config", "-c", default=None, help="Path to override YAML config")
    run_p.set_defaults(func=cmd_run_example)

    bench_p = sub.add_parser("benchmark", help="Run the benchmark suite for a named example")
    bench_p.add_argument("name", help="Example name to benchmark")
    bench_p.set_defaults(func=cmd_benchmark)

    return parser


def app() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    app()
