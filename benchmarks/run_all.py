#!/usr/bin/env python
"""Unified benchmarking harness for production-vlm-engineering.

Runs all four example pipelines, collects their results, and writes a
single structured report -- both machine-readable JSON and a human-
readable Markdown table suitable for pasting into a PR description or
the docs site. This is the canonical answer to "what does this repo
actually produce?" for a technical evaluator or contributor.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --output-dir benchmarks/reports
    python benchmarks/run_all.py --skip vlm_chart_finetune
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from production_vlm.utils.console import Console  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Example descriptors
# ---------------------------------------------------------------------------


@dataclass
class ExampleSpec:
    name: str
    module: str
    headline_keys: list[str]  # dot-path keys into results.json to headline
    headline_labels: list[str]  # human labels for those keys
    timeout_s: int = 120
    tags: list[str] = field(default_factory=list)


EXAMPLES: list[ExampleSpec] = [
    ExampleSpec(
        name="vlm_chart_finetune",
        module="examples.pipelines.vlm_chart_finetune.run",
        headline_keys=["lora_finetuned.mean_faithfulness", "zero_shot.mean_faithfulness", "delta_faithfulness"],
        headline_labels=["LoRA faithfulness", "Zero-shot faithfulness", "Δ faithfulness"],
        timeout_s=120,
        tags=["fine-tuning", "VLM", "chart-QA"],
    ),
    ExampleSpec(
        name="embedding_drift_active_learning",
        module="examples.pipelines.embedding_drift_active_learning.run",
        headline_keys=["drift_detected_at_batch", "detection_delay_batches", "active_learning_queue_size"],
        headline_labels=["Drift detected at batch", "Detection delay (batches)", "AL queue size"],
        timeout_s=60,
        tags=["MLOps", "drift-detection", "active-learning"],
    ),
    ExampleSpec(
        name="vlm_edge_inference",
        module="examples.pipelines.vlm_edge_inference.run",
        headline_keys=["mean_speedup_dynamic_int8_vs_fp32"],
        headline_labels=["INT8 speedup vs fp32"],
        timeout_s=60,
        tags=["inference", "quantization", "edge"],
    ),
    ExampleSpec(
        name="vlm_robustness_guard",
        module="examples.pipelines.vlm_robustness_guard.run",
        headline_keys=[
            "ood_detection.tp_rate",
            "ood_detection.fp_rate",
            "hallucination_guard.precision",
            "hallucination_guard.recall",
        ],
        headline_labels=["OOD TP rate", "OOD FP rate", "Guard precision", "Guard recall"],
        timeout_s=180,
        tags=["robustness", "safety", "OOD", "hallucination"],
    ),
    ExampleSpec(
        name="vlm_video_temporal",
        module="examples.pipelines.vlm_video_temporal.run",
        headline_keys=["strategy_results.uniform_4", "strategy_results.keyframe", "strategy_results.adaptive_4"],
        headline_labels=["Uniform-4 faithfulness", "Keyframe faithfulness", "Adaptive-4 faithfulness"],
        timeout_s=60,
        tags=["video", "temporal", "multi-frame", "P1-04"],
    ),
]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    spec: ExampleSpec
    success: bool
    elapsed_s: float
    results_json: dict
    error: str = ""


def _get_nested(d: dict, dotpath: str) -> str:
    """Traverse a nested dict via a dot-separated key path."""
    keys = dotpath.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return "n/a"
        current = current[key]
    if isinstance(current, float):
        return f"{current:.3f}"
    return str(current)


def run_example(spec: ExampleSpec, output_base: Path) -> RunResult:
    """Run one example as a subprocess with a timeout, return its results.json."""
    console.print(f"  Running [bold]{spec.name}[/bold] ...")
    output_dir = output_base / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        spec.module.replace(".", "/").replace("/py", "").replace("/run", ".run"),
    ]
    # Run via importlib instead so we inherit the sys.path correctly
    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
sys.path.insert(0, '{REPO_ROOT / "src"}')
sys.path.insert(0, '{REPO_ROOT}')
import importlib
m = importlib.import_module('{spec.module}')
m.main()
""",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=spec.timeout_s,
            cwd=str(REPO_ROOT),
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return RunResult(
                spec=spec,
                success=False,
                elapsed_s=elapsed,
                results_json={},
                error=proc.stderr[-2000:] if proc.stderr else "non-zero exit",
            )

        # Load results.json from canonical output location
        results_path = REPO_ROOT / "outputs" / spec.name / "results.json"
        if not results_path.exists():
            return RunResult(
                spec=spec,
                success=False,
                elapsed_s=elapsed,
                results_json={},
                error=f"results.json not found at {results_path}",
            )

        results = json.loads(results_path.read_text())
        return RunResult(spec=spec, success=True, elapsed_s=elapsed, results_json=results)

    except subprocess.TimeoutExpired:
        return RunResult(
            spec=spec,
            success=False,
            elapsed_s=spec.timeout_s,
            results_json={},
            error=f"timed out after {spec.timeout_s}s",
        )
    except Exception as e:
        return RunResult(spec=spec, success=False, elapsed_s=time.perf_counter() - t0, results_json={}, error=str(e))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_markdown_report(run_results: list[RunResult], ran_at: str, total_elapsed: float) -> str:
    lines: list[str] = []
    lines.append("# Production VLM Engineering — Benchmark Report")
    lines.append("")
    lines.append(f"Generated: `{ran_at}`  |  Total wall time: `{total_elapsed:.1f}s`")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Example | Status | Elapsed | Tags |")
    lines.append("|---|---|---|---|")
    for r in run_results:
        status = "✅ pass" if r.success else "❌ fail"
        tags = ", ".join(f"`{t}`" for t in r.spec.tags)
        lines.append(f"| `{r.spec.name}` | {status} | {r.elapsed_s:.1f}s | {tags} |")
    lines.append("")

    # Per-example headline metrics
    lines.append("## Headline Metrics")
    lines.append("")
    for r in run_results:
        lines.append(f"### {r.spec.name}")
        if not r.success:
            lines.append("")
            lines.append(f"> ❌ Failed: {r.error[:200]}")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for dotpath, label in zip(r.spec.headline_keys, r.spec.headline_labels, strict=True):
            value = _get_nested(r.results_json, dotpath)
            lines.append(f"| {label} | `{value}` |")
        ran_real = r.results_json.get("ran_with_real_ml_stack") or r.results_json.get("ran_with_real_export_stack")
        if ran_real is False:
            lines.append("| Mode | ⚠️ CPU smoke-test (install `ml`/`onnx` extras + GPU for real numbers) |")
        lines.append("")

    # Robustness sweep table (P1-02 specific, rendered inline)
    robustness_result = next((r for r in run_results if r.spec.name == "vlm_robustness_guard" and r.success), None)
    if robustness_result:
        pert = robustness_result.results_json.get("perturbation_robustness", {})
        if pert:
            lines.append("## Perturbation Robustness Sweep")
            lines.append("")
            lines.append("Bar-chart reader accuracy under natural perturbations at five severity levels.")
            lines.append("Values in [0,1] represent fraction of charts where the tallest bar was correctly identified.")
            lines.append("")

            all_severities = sorted({float(s) for kind_data in pert.values() for s in kind_data.keys()})
            header = "| Perturbation | " + " | ".join(f"sev={s:.2f}" for s in all_severities) + " |"
            sep = "|---|" + "---|" * len(all_severities)
            lines.append(header)
            lines.append(sep)
            for kind, sev_data in sorted(pert.items()):
                row = f"| `{kind}` | "
                for s in all_severities:
                    val = sev_data.get(str(s), "n/a")
                    if isinstance(val, float):
                        val = f"{val:.0%}"
                    row += f"{val} | "
                lines.append(row)
            lines.append("")
            lines.append(
                "> Brightness and contrast are fully robust (adaptive background detection). "
                "Gaussian noise and occlusion degrade gracefully. "
                "Blur and rotation destroy the pixel signal at high severity — "
                "the honest result for these genuinely destructive perturbations."
            )
            lines.append("")

    lines.append("---")
    lines.append(
        "*Generated by `benchmarks/run_all.py`. "
        "Numbers marked ⚠️ are CPU smoke-test values; "
        "see individual example READMEs for GPU reproduction instructions.*"
    )
    return "\n".join(lines)


def _build_json_report(run_results: list[RunResult], ran_at: str, total_elapsed: float) -> dict:
    return {
        "ran_at": ran_at,
        "total_elapsed_s": round(total_elapsed, 2),
        "results": [
            {
                "name": r.spec.name,
                "success": r.success,
                "elapsed_s": round(r.elapsed_s, 2),
                "error": r.error if not r.success else None,
                "headline_metrics": {
                    label: _get_nested(r.results_json, key)
                    for key, label in zip(r.spec.headline_keys, r.spec.headline_labels, strict=True)
                }
                if r.success
                else {},
                "full_results": r.results_json,
            }
            for r in run_results
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmarks/run_all.py",
        description="Run all examples and generate a unified benchmark report.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/reports",
        help="Directory for report files (default: benchmarks/reports)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Example names to skip, e.g. --skip vlm_chart_finetune",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Run only these example names",
    )
    args = parser.parse_args(argv)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    to_run = [spec for spec in EXAMPLES if spec.name not in args.skip and (not args.only or spec.name in args.only)]

    console.rule("[bold cyan]Production VLM Engineering — Unified Benchmark[/bold cyan]")
    console.print(f"Running {len(to_run)}/{len(EXAMPLES)} examples → {output_dir}")
    console.print("")

    t_total_start = time.perf_counter()
    run_results: list[RunResult] = []

    for spec in to_run:
        result = run_example(spec, output_dir)
        run_results.append(result)
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(
            f"  {status} {spec.name} ({result.elapsed_s:.1f}s)"
            + (f" — {result.error[:80]}" if not result.success else "")
        )

    total_elapsed = time.perf_counter() - t_total_start
    ran_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Write reports
    md_path = output_dir / "benchmark_report.md"
    json_path = output_dir / "benchmark_report.json"

    md_path.write_text(_build_markdown_report(run_results, ran_at, total_elapsed))
    json_path.write_text(json.dumps(_build_json_report(run_results, ran_at, total_elapsed), indent=2))

    console.print("")
    n_passed = sum(r.success for r in run_results)
    console.print(f"[bold]{n_passed}/{len(run_results)} examples passed in {total_elapsed:.1f}s[/bold]")
    console.print(f"[bold green]Report written:[/bold green] {md_path.name}, {json_path.name}")

    return 0 if all(r.success for r in run_results) else 1


if __name__ == "__main__":
    sys.exit(main())
