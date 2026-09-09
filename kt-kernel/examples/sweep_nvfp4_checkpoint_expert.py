"""Tune checkpoint-native NVFP4 decode settings with a staged sweep."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


NVFP4_ENV_KEYS = (
    "KT_NVFP4_BLOCKED_LAYOUT",
    "KT_NVFP4_INTERLEAVED_LAYOUT",
    "KT_NVFP4_QUARTET_LAYOUT",
    "KT_NVFP4_DECODE_TILE_BATCH",
    "KT_NVFP4_PREFETCH_GROUPS",
    "KT_NVFP4_N_BLOCK",
    "KT_NVFP4_BF16_SCALES",
    "KT_NVFP4_BATCH_SCALE_DECODE",
    "KT_NVFP4_VBMI_DECODE",
    "KT_NVFP4_DIRECT_DOWN_INPUT",
    "KT_NVFP4_DIRECT_BF16_OUTPUT",
    "KT_NVFP4_STATIC_SCHEDULE",
    "KT_NVFP4_ADAPTIVE_SCHEDULE",
    "KT_NVFP4_STATIC_GATE_UP",
    "KT_NVFP4_STATIC_DOWN",
    "KT_NVFP4_ZERO_CPU_FASTPATH",
)

DEFAULT_CPU_TOP_K_VALUES = tuple(range(9))
DEFAULT_CPU_TOP_K_WEIGHTS = (
    448,
    2247,
    6196,
    10953,
    14184,
    14814,
    12692,
    8910,
    4556,
)

# Start from the strongest settings found in the GLM-5.3 tuning runs. Every
# value is still retested, so this only reduces the path to a good combination.
TUNED_START = {
    "KT_NVFP4_BLOCKED_LAYOUT": "1",
    "KT_NVFP4_INTERLEAVED_LAYOUT": "0",
    "KT_NVFP4_QUARTET_LAYOUT": "1",
    "KT_NVFP4_DECODE_TILE_BATCH": "4",
    "KT_NVFP4_PREFETCH_GROUPS": "3",
    "KT_NVFP4_N_BLOCK": "256",
    "KT_NVFP4_BF16_SCALES": "1",
    "KT_NVFP4_BATCH_SCALE_DECODE": "0",
    "KT_NVFP4_VBMI_DECODE": "1",
    "KT_NVFP4_DIRECT_DOWN_INPUT": "1",
    "KT_NVFP4_DIRECT_BF16_OUTPUT": "1",
    "KT_NVFP4_STATIC_SCHEDULE": "0",
    "KT_NVFP4_ADAPTIVE_SCHEDULE": "1",
    "KT_NVFP4_STATIC_GATE_UP": "0",
    "KT_NVFP4_STATIC_DOWN": "1",
    "KT_NVFP4_ZERO_CPU_FASTPATH": "1",
}

STAGES = (
    (
        "layout",
        (
            (
                "row-major",
                {
                    "KT_NVFP4_BLOCKED_LAYOUT": "0",
                    "KT_NVFP4_INTERLEAVED_LAYOUT": "0",
                    "KT_NVFP4_QUARTET_LAYOUT": "0",
                },
            ),
            (
                "blocked-n16",
                {
                    "KT_NVFP4_BLOCKED_LAYOUT": "1",
                    "KT_NVFP4_INTERLEAVED_LAYOUT": "0",
                    "KT_NVFP4_QUARTET_LAYOUT": "0",
                },
            ),
            (
                "interleaved-n16",
                {
                    "KT_NVFP4_BLOCKED_LAYOUT": "1",
                    "KT_NVFP4_INTERLEAVED_LAYOUT": "1",
                    "KT_NVFP4_QUARTET_LAYOUT": "0",
                },
            ),
            (
                "quartet-n64",
                {
                    "KT_NVFP4_BLOCKED_LAYOUT": "1",
                    "KT_NVFP4_INTERLEAVED_LAYOUT": "0",
                    "KT_NVFP4_QUARTET_LAYOUT": "1",
                },
            ),
        ),
    ),
    (
        "scales",
        (
            (
                "e4m3",
                {
                    "KT_NVFP4_BF16_SCALES": "0",
                    "KT_NVFP4_BATCH_SCALE_DECODE": "0",
                },
            ),
            (
                "e4m3-batched",
                {
                    "KT_NVFP4_BF16_SCALES": "0",
                    "KT_NVFP4_BATCH_SCALE_DECODE": "1",
                },
            ),
            (
                "bf16",
                {
                    "KT_NVFP4_BF16_SCALES": "1",
                    "KT_NVFP4_BATCH_SCALE_DECODE": "0",
                },
            ),
        ),
    ),
    (
        "weight-decode",
        (
            ("unpack", {"KT_NVFP4_VBMI_DECODE": "0"}),
            ("vbmi", {"KT_NVFP4_VBMI_DECODE": "1"}),
        ),
    ),
    (
        "tile-batch",
        tuple(
            (str(value), {"KT_NVFP4_DECODE_TILE_BATCH": str(value)})
            for value in (1, 2, 4)
        ),
    ),
    (
        "prefetch",
        tuple(
            (str(value), {"KT_NVFP4_PREFETCH_GROUPS": str(value)})
            for value in (0, 2, 3, 5, 8)
        ),
    ),
    (
        "n-block",
        tuple(
            (str(value), {"KT_NVFP4_N_BLOCK": str(value)})
            for value in (64, 96, 128, 160, 192, 224, 256)
        ),
    ),
    (
        "direct-down-input",
        tuple(
            (str(value), {"KT_NVFP4_DIRECT_DOWN_INPUT": str(value)}) for value in (0, 1)
        ),
    ),
    (
        "direct-bf16-output",
        tuple(
            (str(value), {"KT_NVFP4_DIRECT_BF16_OUTPUT": str(value)})
            for value in (0, 1)
        ),
    ),
    (
        "schedule",
        (
            (
                "dynamic",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "0",
                    "KT_NVFP4_STATIC_GATE_UP": "0",
                    "KT_NVFP4_STATIC_DOWN": "0",
                },
            ),
            (
                "dynamic-static-down",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "0",
                    "KT_NVFP4_STATIC_GATE_UP": "0",
                    "KT_NVFP4_STATIC_DOWN": "1",
                },
            ),
            (
                "adaptive",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "1",
                    "KT_NVFP4_STATIC_GATE_UP": "0",
                    "KT_NVFP4_STATIC_DOWN": "0",
                },
            ),
            (
                "adaptive-static-down",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "1",
                    "KT_NVFP4_STATIC_GATE_UP": "0",
                    "KT_NVFP4_STATIC_DOWN": "1",
                },
            ),
            (
                "static-gate-up",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "0",
                    "KT_NVFP4_STATIC_GATE_UP": "1",
                    "KT_NVFP4_STATIC_DOWN": "0",
                },
            ),
            (
                "static-gate-up-down",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "0",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "0",
                    "KT_NVFP4_STATIC_GATE_UP": "1",
                    "KT_NVFP4_STATIC_DOWN": "1",
                },
            ),
            (
                "static-all",
                {
                    "KT_NVFP4_STATIC_SCHEDULE": "1",
                    "KT_NVFP4_ADAPTIVE_SCHEDULE": "0",
                    "KT_NVFP4_STATIC_GATE_UP": "0",
                    "KT_NVFP4_STATIC_DOWN": "0",
                },
            ),
        ),
    ),
)

BENCHMARK_RE = re.compile(
    r"benchmark \([^\n]*cpu_top_k=(?P<cpu_top_k>[0-9]+)[^\n]*\):\s+"
    r"(?P<ms>[0-9.]+) ms/forward\s+"
    r"(?P<forwards>[0-9.]+) forwards/s\s+"
    r"(?P<checkpoint>[0-9.]+) "
    r"(?:checkpoint_weight|effective_weight)_GB/s"
    r"(?:\s+(?P<resident>[0-9.]+) resident_weight_GB/s)?"
)
SAMPLES_RE = re.compile(
    r"samples_ms=\[(?P<samples>[^]]+)]\s+spread=(?P<spread>[0-9.]+)%"
)


@dataclass
class RunResult:
    run_id: str
    pass_number: int
    stage: str
    candidate: str
    config: dict[str, str]
    returncode: int
    status: str
    median_ms: float | None
    raw_weighted_ms: float | None
    cpu_top_k_ms: dict[str, float]
    forwards_per_second: float | None
    checkpoint_gbps: float | None
    resident_gbps: float | None
    sample_ms: list[float]
    spread_percent: float | None
    wall_seconds: float
    log_path: str

    def as_dict(self) -> dict[str, object]:
        return vars(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "RunResult":
        values.setdefault("raw_weighted_ms", values.get("median_ms"))
        values.setdefault("cpu_top_k_ms", {})
        return cls(**values)  # type: ignore[arg-type]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coordinate-sweep the recent checkpoint-native NVFP4 decode "
            "optimizations. Each candidate runs in a fresh process."
        )
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=31)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--numa-nodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--benchmark-expert-count", type=int, default=32)
    parser.add_argument("--route-top-k", type=int, default=8)
    parser.add_argument(
        "--cpu-top-k",
        type=int,
        help=(
            "Benchmark one CPU top-k value. This compatibility shorthand "
            "overrides --cpu-top-k-values and --cpu-top-k-weights."
        ),
    )
    parser.add_argument(
        "--cpu-top-k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_CPU_TOP_K_VALUES),
        help="CPU route counts measured for every candidate.",
    )
    parser.add_argument(
        "--cpu-top-k-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_CPU_TOP_K_WEIGHTS),
        help=(
            "Observed frequency for each --cpu-top-k-values entry. Defaults "
            "to the GLM-5.3 frequency-global profile."
        ),
    )
    parser.add_argument(
        "--gpu-floor-ms",
        type=float,
        default=0.45,
        help=(
            "Hybrid GPU latency floor. Candidate score is the weighted mean "
            "of max(CPU latency, this floor)."
        ),
    )
    parser.add_argument("--benchmark-warmup", type=int, default=32)
    parser.add_argument("--benchmark-iterations", type=int, default=512)
    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--cooldown-seconds", type=float, default=40.0)
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Coordinate-descent passes; two catches most knob interactions.",
    )
    parser.add_argument(
        "--minimum-improvement-percent",
        type=float,
        default=0.5,
        help="Keep the current value unless the candidate wins by this much.",
    )
    parser.add_argument(
        "--paired-confirmation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remeasure the prior and provisional winning configurations.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(stage for stage, _ in STAGES),
        help="Run only selected stages, in their normal order.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260908)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory; defaults to nvfp4-sweep-TIMESTAMP.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed run IDs in an existing --output-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned candidates without running the benchmark.",
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path(__file__).with_name("validate_nvfp4_checkpoint_expert.py"),
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.passes < 1:
        parser.error("--passes must be at least 1")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    if args.cpu_top_k is not None:
        args.cpu_top_k_values = [args.cpu_top_k]
        args.cpu_top_k_weights = [1.0]
    if len(args.cpu_top_k_values) != len(args.cpu_top_k_weights):
        parser.error(
            "--cpu-top-k-values and --cpu-top-k-weights must have equal length"
        )
    combined_weights = {}
    for cpu_top_k, weight in zip(args.cpu_top_k_values, args.cpu_top_k_weights):
        combined_weights[cpu_top_k] = combined_weights.get(cpu_top_k, 0.0) + weight
    args.cpu_top_k_values = list(combined_weights)
    args.cpu_top_k_weights = list(combined_weights.values())
    if not args.cpu_top_k_values:
        parser.error("at least one --cpu-top-k-values entry is required")
    if any(weight < 0 for weight in args.cpu_top_k_weights):
        parser.error("--cpu-top-k-weights cannot contain negative values")
    if sum(args.cpu_top_k_weights) <= 0:
        parser.error("--cpu-top-k-weights must have a positive sum")
    for cpu_top_k in args.cpu_top_k_values:
        if not 0 <= cpu_top_k <= args.route_top_k:
            parser.error("CPU top-k values must be between zero and --route-top-k")
        if cpu_top_k > args.benchmark_expert_count:
            parser.error("CPU top-k values cannot exceed --benchmark-expert-count")
    if args.gpu_floor_ms < 0:
        parser.error("--gpu-floor-ms cannot be negative")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    return args


def config_key(config: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((key, config[key]) for key in NVFP4_ENV_KEYS)


def candidate_configs(
    current: dict[str, str],
    options: tuple[tuple[str, dict[str, str]], ...],
) -> list[tuple[str, dict[str, str]]]:
    result = []
    seen = set()
    for label, updates in options:
        config = current | updates
        key = config_key(config)
        if key in seen:
            continue
        seen.add(key)
        result.append((label, config))
    return result


def benchmark_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(args.validator),
        str(args.model_path),
        "--layer",
        str(args.layer),
        "--expert",
        str(args.expert),
        "--hidden-size",
        str(args.hidden_size),
        "--intermediate-size",
        str(args.intermediate_size),
        "--threads",
        str(args.threads),
        "--numa-nodes",
        *(str(node) for node in args.numa_nodes),
        "--backend",
        "amx",
        "--benchmark-expert-count",
        str(args.benchmark_expert_count),
        "--benchmark-top-k",
        str(args.route_top_k),
        "--benchmark-cpu-top-k-values",
        *(str(value) for value in args.cpu_top_k_values),
        "--benchmark-warmup",
        str(args.benchmark_warmup),
        "--benchmark-iterations",
        str(args.benchmark_iterations),
        "--benchmark-repeats",
        str(args.benchmark_repeats),
    ]


def parse_result(
    *,
    run_id: str,
    pass_number: int,
    stage: str,
    candidate: str,
    config: dict[str, str],
    returncode: int,
    output: str,
    wall_seconds: float,
    log_path: Path,
    cpu_top_k_values: list[int],
    cpu_top_k_weights: list[float],
    gpu_floor_ms: float,
) -> RunResult:
    benchmark_matches = list(BENCHMARK_RE.finditer(output))
    sample_matches = list(SAMPLES_RE.finditer(output))
    passed = "AMXFP4_KGroup_MOE:" in output and "status=PASS" in output
    parsed = {
        int(match.group("cpu_top_k")): match.groupdict() for match in benchmark_matches
    }
    missing = set(cpu_top_k_values) - set(parsed)
    status = (
        "ok"
        if returncode == 0 and passed and benchmark_matches and not missing
        else "failed"
    )

    cpu_top_k_ms = {
        str(cpu_top_k): float(parsed[cpu_top_k]["ms"])
        for cpu_top_k in cpu_top_k_values
        if cpu_top_k in parsed
    }
    if status == "ok":
        total_weight = sum(cpu_top_k_weights)
        raw_weighted_ms = (
            sum(
                weight * cpu_top_k_ms[str(cpu_top_k)]
                for cpu_top_k, weight in zip(cpu_top_k_values, cpu_top_k_weights)
            )
            / total_weight
        )
        median_ms = (
            sum(
                weight * max(cpu_top_k_ms[str(cpu_top_k)], gpu_floor_ms)
                for cpu_top_k, weight in zip(cpu_top_k_values, cpu_top_k_weights)
            )
            / total_weight
        )
        forwards = 1000.0 / median_ms
        checkpoint_gbps = (
            sum(
                weight * float(parsed[cpu_top_k]["checkpoint"])
                for cpu_top_k, weight in zip(cpu_top_k_values, cpu_top_k_weights)
            )
            / total_weight
        )
        resident_values = [
            (weight, parsed[cpu_top_k]["resident"])
            for cpu_top_k, weight in zip(cpu_top_k_values, cpu_top_k_weights)
        ]
        resident_gbps = (
            sum(weight * float(value) for weight, value in resident_values)
            / total_weight
            if all(value is not None for _, value in resident_values)
            else None
        )
    else:
        median_ms = raw_weighted_ms = None
        forwards = checkpoint_gbps = resident_gbps = None

    sample_ms = [
        float(value.strip())
        for match in sample_matches
        for value in match.group("samples").split(",")
    ]
    spread_percent = (
        max(float(match.group("spread")) for match in sample_matches)
        if sample_matches
        else None
    )

    return RunResult(
        run_id=run_id,
        pass_number=pass_number,
        stage=stage,
        candidate=candidate,
        config=config,
        returncode=returncode,
        status=status,
        median_ms=median_ms,
        raw_weighted_ms=raw_weighted_ms,
        cpu_top_k_ms=cpu_top_k_ms,
        forwards_per_second=forwards,
        checkpoint_gbps=checkpoint_gbps,
        resident_gbps=resident_gbps,
        sample_ms=sample_ms,
        spread_percent=spread_percent,
        wall_seconds=wall_seconds,
        log_path=str(log_path),
    )


def append_result(path: Path, result: RunResult) -> None:
    with path.open("a", encoding="utf-8") as output:
        json.dump(result.as_dict(), output, sort_keys=True)
        output.write("\n")


def load_results(path: Path) -> dict[str, RunResult]:
    if not path.exists():
        return {}
    results = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                result = RunResult.from_dict(json.loads(line))
                results[result.run_id] = result
    return results


def write_best_env(path: Path, config: dict[str, str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by sweep_nvfp4_checkpoint_expert.py",
        "",
    ]
    lines.extend(f"export {key}={shlex.quote(config[key])}" for key in NVFP4_ENV_KEYS)
    lines.extend(
        [
            "",
            "# End-to-end hybrid-path settings established separately.",
            "export KT_CPU_EXPERT_IDS_INT32=1",
            "export SGLANG_KT_HYBRID_DIRECT_CPU_INPUT=1",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_candidate(
    args: argparse.Namespace,
    output_dir: Path,
    results_path: Path,
    completed: dict[str, RunResult],
    run_id: str,
    pass_number: int,
    stage: str,
    candidate: str,
    config: dict[str, str],
    command: list[str],
) -> RunResult | None:
    if run_id in completed:
        result = completed[run_id]
        print(
            f"[{run_id}] resume {result.status}: "
            f"{result.median_ms if result.median_ms is not None else 'n/a'} ms"
        )
        return result

    print(f"\n[{run_id}] {stage}={candidate}")
    print("  " + " ".join(f"{key}={config[key]}" for key in NVFP4_ENV_KEYS))
    if args.dry_run:
        return None

    env = os.environ.copy()
    for key in NVFP4_ENV_KEYS:
        env.pop(key, None)
    env.update(config)

    started = time.perf_counter()
    process = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    log_path = output_dir / f"{run_id}.log"
    log_path.write_text(process.stdout, encoding="utf-8")
    result = parse_result(
        run_id=run_id,
        pass_number=pass_number,
        stage=stage,
        candidate=candidate,
        config=config,
        returncode=process.returncode,
        output=process.stdout,
        wall_seconds=wall_seconds,
        log_path=log_path,
        cpu_top_k_values=args.cpu_top_k_values,
        cpu_top_k_weights=args.cpu_top_k_weights,
        gpu_floor_ms=args.gpu_floor_ms,
    )
    append_result(results_path, result)
    completed[run_id] = result

    if result.status == "ok":
        samples = (
            f", spread={result.spread_percent:.2f}%"
            if result.spread_percent is not None
            else ""
        )
        print(
            f"  result: hybrid score={result.median_ms:.3f} ms, "
            f"raw weighted={result.raw_weighted_ms:.3f} ms, "
            f"cpu_top_k_ms={result.cpu_top_k_ms}{samples}"
        )
    else:
        print(
            f"  FAILED (exit={result.returncode}); see {result.log_path}",
            file=sys.stderr,
        )
    return result


def cool_down(seconds: float, next_run: int, total_runs: int) -> None:
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(
            f"\r  cooling before run {next_run}/{total_runs}: "
            f"{remaining:5.1f}s remaining",
            end="",
            flush=True,
        )
        time.sleep(min(1.0, remaining))
    print("\r" + " " * 70 + "\r", end="", flush=True)


def count_planned_runs(selected_stages: list[tuple[str, tuple]]) -> int:
    # Options inside one stage are unique for all supplied starting configs.
    return sum(len(options) for _, options in selected_stages)


def print_ranking(results: list[RunResult]) -> None:
    successful = [
        result
        for result in results
        if result.status == "ok" and result.median_ms is not None
    ]
    if not successful:
        print("  no successful candidates")
        return
    for rank, result in enumerate(
        sorted(successful, key=lambda item: item.median_ms)[:5], 1
    ):
        print(
            f"  {rank}. {result.candidate:<24} "
            f"{result.median_ms:.3f} ms "
            f"({result.forwards_per_second:.1f} forwards/s)"
        )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path(f"nvfp4-sweep-{timestamp}")
    output_dir = output_dir.resolve()
    results_path = output_dir / "results.jsonl"
    metadata_path = output_dir / "metadata.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n"
            "Pass --resume or select another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not metadata_path.exists():
            raise SystemExit(f"Cannot resume without metadata: {metadata_path}")
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior_metadata.get("schema_version") != 2:
            raise SystemExit(
                "This result directory uses the earlier single-route score. "
                "Start a new sweep directory for weighted hybrid scoring."
            )

    selected_names = set(args.stages) if args.stages else None
    selected_stages = [
        (name, options)
        for name, options in STAGES
        if selected_names is None or name in selected_names
    ]
    command = benchmark_command(args)
    completed = load_results(results_path) if args.resume else {}
    current = TUNED_START.copy()
    initial = current.copy()
    all_stage_winners = []
    run_number = 0
    primary_runs = count_planned_runs(selected_stages) * args.passes
    confirmation_runs = (
        2 * len(selected_stages) * args.passes if args.paired_confirmation else 0
    )
    maximum_runs = primary_runs + confirmation_runs

    metadata = {
        "schema_version": 2,
        "model_path": str(args.model_path.resolve()),
        "command": command,
        "cooldown_seconds": args.cooldown_seconds,
        "passes": args.passes,
        "minimum_improvement_percent": args.minimum_improvement_percent,
        "paired_confirmation": args.paired_confirmation,
        "cpu_top_k_values": args.cpu_top_k_values,
        "cpu_top_k_weights": args.cpu_top_k_weights,
        "gpu_floor_ms": args.gpu_floor_ms,
        "stages": [name for name, _ in selected_stages],
        "tuned_start": initial,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Results: {output_dir}")
    print(f"Benchmark: {shlex.join(command)}")
    print(
        f"Plan: {primary_runs} candidates plus up to "
        f"{confirmation_runs} confirmation runs, {args.passes} pass(es), "
        f"{args.cooldown_seconds:g}s cooldown"
    )
    print(
        "Hybrid score: weighted mean max(cpu_top_k_ms, "
        f"{args.gpu_floor_ms:g} ms GPU floor)"
    )

    for pass_number in range(1, args.passes + 1):
        print(f"\n{'=' * 24} PASS {pass_number} {'=' * 24}")
        for stage_index, (stage, options) in enumerate(selected_stages, 1):
            candidates = candidate_configs(current, options)
            random.Random(args.shuffle_seed + pass_number * 1000 + stage_index).shuffle(
                candidates
            )
            stage_results = []

            for candidate_index, (candidate, config) in enumerate(candidates, 1):
                run_number += 1
                run_id = (
                    f"p{pass_number:02d}-{stage_index:02d}-{stage}-"
                    f"{candidate_index:02d}-{candidate}"
                )
                was_completed = run_id in completed
                result = run_candidate(
                    args,
                    output_dir,
                    results_path,
                    completed,
                    run_id,
                    pass_number,
                    stage,
                    candidate,
                    config,
                    command,
                )
                if result is not None:
                    stage_results.append(result)

                if not args.dry_run and not was_completed:
                    cool_down(
                        args.cooldown_seconds,
                        run_number + 1,
                        maximum_runs,
                    )

            if args.dry_run:
                continue

            print(f"\n{stage} ranking:")
            print_ranking(stage_results)
            successful = [
                result
                for result in stage_results
                if result.status == "ok" and result.median_ms is not None
            ]
            if not successful:
                print(f"Keeping prior {stage} configuration after failures.")
                continue

            winner = min(successful, key=lambda item: item.median_ms)
            current_result = next(
                (
                    result
                    for result in successful
                    if config_key(result.config) == config_key(current)
                ),
                None,
            )
            if current_result is None:
                current = winner.config.copy()
                decision = "accepted; prior combination did not complete"
            else:
                improvement = (
                    (current_result.median_ms - winner.median_ms)
                    / current_result.median_ms
                    * 100.0
                )
                changed = config_key(winner.config) != config_key(current)
                worth_confirming = (
                    changed
                    and improvement >= args.minimum_improvement_percent
                    and args.paired_confirmation
                )
                if worth_confirming:
                    result_positions = {
                        result.run_id: index
                        for index, result in enumerate(stage_results)
                    }
                    if (
                        result_positions[current_result.run_id]
                        < result_positions[winner.run_id]
                    ):
                        confirmation_order = (
                            ("winner", winner),
                            ("current", current_result),
                        )
                    else:
                        confirmation_order = (
                            ("current", current_result),
                            ("winner", winner),
                        )

                    confirmations = {}
                    print("Paired confirmation in reverse measurement order:")
                    for role, source_result in confirmation_order:
                        run_number += 1
                        run_id = (
                            f"p{pass_number:02d}-{stage_index:02d}-{stage}-"
                            f"confirm-{role}"
                        )
                        was_completed = run_id in completed
                        confirmation = run_candidate(
                            args,
                            output_dir,
                            results_path,
                            completed,
                            run_id,
                            pass_number,
                            stage,
                            f"confirm-{role}-{source_result.candidate}",
                            source_result.config,
                            command,
                        )
                        if confirmation is not None:
                            confirmations[role] = confirmation
                        if not was_completed:
                            cool_down(
                                args.cooldown_seconds,
                                run_number + 1,
                                maximum_runs,
                            )

                    confirmed_current = confirmations.get("current")
                    confirmed_winner = confirmations.get("winner")
                    if (
                        confirmed_current is not None
                        and confirmed_current.status == "ok"
                        and confirmed_current.median_ms is not None
                        and confirmed_winner is not None
                        and confirmed_winner.status == "ok"
                        and confirmed_winner.median_ms is not None
                    ):
                        current_score = statistics.median(
                            [
                                current_result.median_ms,
                                confirmed_current.median_ms,
                            ]
                        )
                        winner_score = statistics.median(
                            [winner.median_ms, confirmed_winner.median_ms]
                        )
                        confirmed_improvement = (
                            (current_score - winner_score) / current_score * 100.0
                        )
                        if confirmed_improvement >= args.minimum_improvement_percent:
                            current = winner.config.copy()
                            decision = (
                                "accepted after paired confirmation; "
                                f"{confirmed_improvement:.2f}% faster"
                            )
                        else:
                            decision = (
                                "kept prior combination; paired change was "
                                f"{confirmed_improvement:.2f}%"
                            )
                    else:
                        decision = "kept prior combination; paired confirmation failed"
                elif changed and improvement >= args.minimum_improvement_percent:
                    current = winner.config.copy()
                    decision = f"accepted; {improvement:.2f}% faster"
                else:
                    decision = (
                        "kept prior combination; best measured change "
                        f"was {improvement:.2f}%"
                    )
            all_stage_winners.append(winner)
            write_best_env(output_dir / "best-env.sh", current)
            print(f"Decision: {winner.candidate}: {decision}")

    if args.dry_run:
        print("\nDry run complete; no benchmarks or cooldowns were performed.")
        return

    write_best_env(output_dir / "best-env.sh", current)
    summary = {
        "initial_config": initial,
        "best_config": current,
        "successful_runs": sum(result.status == "ok" for result in completed.values()),
        "failed_runs": sum(result.status != "ok" for result in completed.values()),
        "stage_winners": [result.as_dict() for result in all_stage_winners],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 22} FINAL SETTINGS {'=' * 22}")
    for key in NVFP4_ENV_KEYS:
        print(f"export {key}={current[key]}")
    print(f"\nSourceable settings: {output_dir / 'best-env.sh'}")
    print(f"Complete results:    {results_path}")


if __name__ == "__main__":
    main()
