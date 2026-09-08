"""Tune checkpoint-native NVFP4 decode settings with a staged sweep."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
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
        tuple((str(value), {"KT_NVFP4_DECODE_TILE_BATCH": str(value)}) for value in (1, 2, 4)),
    ),
    (
        "prefetch",
        tuple((str(value), {"KT_NVFP4_PREFETCH_GROUPS": str(value)}) for value in (0, 2, 3, 5, 8)),
    ),
    (
        "n-block",
        tuple((str(value), {"KT_NVFP4_N_BLOCK": str(value)}) for value in (64, 96, 128, 160, 192, 224, 256)),
    ),
    (
        "direct-down-input",
        tuple((str(value), {"KT_NVFP4_DIRECT_DOWN_INPUT": str(value)}) for value in (0, 1)),
    ),
    (
        "direct-bf16-output",
        tuple((str(value), {"KT_NVFP4_DIRECT_BF16_OUTPUT": str(value)}) for value in (0, 1)),
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
    r"benchmark \([^\n]+\):\s+"
    r"(?P<ms>[0-9.]+) ms/forward\s+"
    r"(?P<forwards>[0-9.]+) forwards/s\s+"
    r"(?P<checkpoint>[0-9.]+) "
    r"(?:checkpoint_weight|effective_weight)_GB/s"
    r"(?:\s+(?P<resident>[0-9.]+) resident_weight_GB/s)?"
)
SAMPLES_RE = re.compile(r"samples_ms=\[(?P<samples>[^]]+)]\s+spread=(?P<spread>[0-9.]+)%")


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
    parser.add_argument("--cpu-top-k", type=int, default=7)
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
    if not 0 <= args.cpu_top_k <= args.route_top_k:
        parser.error("--cpu-top-k must be between zero and --route-top-k")
    if args.cpu_top_k > args.benchmark_expert_count:
        parser.error("--cpu-top-k cannot exceed --benchmark-expert-count")
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
        "--benchmark-cpu-top-k",
        str(args.cpu_top_k),
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
) -> RunResult:
    benchmark_matches = list(BENCHMARK_RE.finditer(output))
    sample_match = SAMPLES_RE.search(output)
    passed = "AMXFP4_KGroup_MOE:" in output and "status=PASS" in output
    status = "ok" if returncode == 0 and passed and benchmark_matches else "failed"

    if benchmark_matches:
        values = benchmark_matches[-1].groupdict()
        median_ms = float(values["ms"])
        forwards = float(values["forwards"])
        checkpoint_gbps = float(values["checkpoint"])
        resident_gbps = float(values["resident"]) if values["resident"] else None
    else:
        median_ms = forwards = checkpoint_gbps = resident_gbps = None

    if sample_match:
        sample_ms = [float(value.strip()) for value in sample_match.group("samples").split(",")]
        spread_percent = float(sample_match.group("spread"))
    else:
        sample_ms = []
        spread_percent = None

    return RunResult(
        run_id=run_id,
        pass_number=pass_number,
        stage=stage,
        candidate=candidate,
        config=config,
        returncode=returncode,
        status=status,
        median_ms=median_ms,
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
            f"[{run_id}] resume {result.status}: " f"{result.median_ms if result.median_ms is not None else 'n/a'} ms"
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
    )
    append_result(results_path, result)
    completed[run_id] = result

    if result.status == "ok":
        samples = f", spread={result.spread_percent:.2f}%" if result.spread_percent is not None else ""
        print(f"  result: {result.median_ms:.3f} ms/forward, " f"{result.forwards_per_second:.1f} forwards/s{samples}")
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
            f"\r  cooling before run {next_run}/{total_runs}: " f"{remaining:5.1f}s remaining",
            end="",
            flush=True,
        )
        time.sleep(min(1.0, remaining))
    print("\r" + " " * 70 + "\r", end="", flush=True)


def count_planned_runs(selected_stages: list[tuple[str, tuple]]) -> int:
    # Options inside one stage are unique for all supplied starting configs.
    return sum(len(options) for _, options in selected_stages)


def print_ranking(results: list[RunResult]) -> None:
    successful = [result for result in results if result.status == "ok" and result.median_ms is not None]
    if not successful:
        print("  no successful candidates")
        return
    for rank, result in enumerate(sorted(successful, key=lambda item: item.median_ms)[:5], 1):
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

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}\n" "Pass --resume or select another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = set(args.stages) if args.stages else None
    selected_stages = [(name, options) for name, options in STAGES if selected_names is None or name in selected_names]
    command = benchmark_command(args)
    completed = load_results(results_path) if args.resume else {}
    current = TUNED_START.copy()
    initial = current.copy()
    all_stage_winners = []
    run_number = 0
    total_runs = count_planned_runs(selected_stages) * args.passes

    metadata = {
        "model_path": str(args.model_path.resolve()),
        "command": command,
        "cooldown_seconds": args.cooldown_seconds,
        "passes": args.passes,
        "minimum_improvement_percent": args.minimum_improvement_percent,
        "stages": [name for name, _ in selected_stages],
        "tuned_start": initial,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Results: {output_dir}")
    print(f"Benchmark: {shlex.join(command)}")
    print(f"Plan: {total_runs} candidates, {args.passes} pass(es), " f"{args.cooldown_seconds:g}s cooldown")

    for pass_number in range(1, args.passes + 1):
        print(f"\n{'=' * 24} PASS {pass_number} {'=' * 24}")
        for stage_index, (stage, options) in enumerate(selected_stages, 1):
            candidates = candidate_configs(current, options)
            random.Random(args.shuffle_seed + pass_number * 1000 + stage_index).shuffle(candidates)
            stage_results = []

            for candidate_index, (candidate, config) in enumerate(candidates, 1):
                run_number += 1
                run_id = f"p{pass_number:02d}-{stage_index:02d}-{stage}-" f"{candidate_index:02d}-{candidate}"
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

                if run_number < total_runs and not args.dry_run and not was_completed:
                    cool_down(
                        args.cooldown_seconds,
                        run_number + 1,
                        total_runs,
                    )

            if args.dry_run:
                continue

            print(f"\n{stage} ranking:")
            print_ranking(stage_results)
            successful = [result for result in stage_results if result.status == "ok" and result.median_ms is not None]
            if not successful:
                print(f"Keeping prior {stage} configuration after failures.")
                continue

            winner = min(successful, key=lambda item: item.median_ms)
            current_result = next(
                (result for result in successful if config_key(result.config) == config_key(current)),
                None,
            )
            if current_result is None:
                current = winner.config.copy()
                decision = "accepted; prior combination did not complete"
            else:
                improvement = (current_result.median_ms - winner.median_ms) / current_result.median_ms * 100.0
                if config_key(winner.config) != config_key(current) and improvement >= args.minimum_improvement_percent:
                    current = winner.config.copy()
                    decision = f"accepted; {improvement:.2f}% faster"
                else:
                    decision = "kept prior combination; best measured change " f"was {improvement:.2f}%"
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
