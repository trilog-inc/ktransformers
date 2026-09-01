"""Compare a real checkpoint NVFP4 expert with a PyTorch reference."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from kt_kernel import kt_kernel_ext
from kt_kernel.utils.loader import NVFP4SafeTensorLoader


E2M1 = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=31)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=26)
    parser.add_argument("--numa-nodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--backend",
        choices=("all", "amx", "avx2"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--relative-tolerance", type=float, default=0.08)
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument(
        "--benchmark-expert-count",
        type=int,
        default=1,
        help=(
            "Load this many consecutive experts starting at --expert and rotate "
            "them during timing. Values larger than one measure a streaming "
            "working set instead of repeatedly hitting one expert in LLC."
        ),
    )
    parser.add_argument(
        "--benchmark-top-k",
        type=int,
        default=1,
        help=(
            "Route this many experts in each timed forward. The expert count "
            "must be divisible by top-k so timing can rotate disjoint banks."
        ),
    )
    return parser.parse_args()


def make_cpu_infer(threads: int, numa_nodes: list[int]):
    if threads < len(numa_nodes):
        raise ValueError("--threads must be at least the number of NUMA nodes")
    counts = [threads // len(numa_nodes)] * len(numa_nodes)
    for index in range(threads % len(numa_nodes)):
        counts[index] += 1

    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = len(numa_nodes)
    config.subpool_numa_map = numa_nodes
    config.subpool_thread_count = counts
    return kt_kernel_ext.CPUInfer(config)


def load_checkpoint_experts(args: argparse.Namespace):
    if args.benchmark_expert_count < 1:
        raise ValueError("--benchmark-expert-count must be at least 1")
    if args.benchmark_top_k < 1:
        raise ValueError("--benchmark-top-k must be at least 1")
    if args.benchmark_expert_count % args.benchmark_top_k:
        raise ValueError(
            "--benchmark-expert-count must be divisible by --benchmark-top-k"
        )
    expert_ids = list(
        range(args.expert, args.expert + args.benchmark_expert_count)
    )
    loader = NVFP4SafeTensorLoader(str(args.model_path))
    weights = loader.load_experts(
        f"model.layers.{args.layer}", expert_ids=expert_ids
    )
    return expert_ids, [
        {name: values[expert_id] for name, values in weights.items()}
        for expert_id in expert_ids
    ]


def dequantize_projection(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
) -> torch.Tensor:
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    unpacked = torch.empty((rows, columns), dtype=torch.uint8)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = packed >> 4
    values = E2M1[unpacked.long()].view(rows, columns // 16, 16)
    result = values * block_scale.float().unsqueeze(-1)
    result.mul_(float(tensor_scale.item()))
    return result.view(rows, columns)


def reference_forward(
    inputs: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    gate_weight = dequantize_projection(
        weights["gate"], weights["gate_scale"], weights["gate_scale_2"]
    )
    gate = torch.mm(inputs.float(), gate_weight.t()).to(torch.bfloat16)
    del gate_weight
    gc.collect()

    up_weight = dequantize_projection(
        weights["up"], weights["up_scale"], weights["up_scale_2"]
    )
    up = torch.mm(inputs.float(), up_weight.t()).to(torch.bfloat16)
    del up_weight
    gc.collect()

    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
        torch.bfloat16
    )
    down_weight = dequantize_projection(
        weights["down"], weights["down_scale"], weights["down_scale_2"]
    )
    output = torch.mm(activated.float(), down_weight.t()).to(torch.bfloat16)
    del down_weight
    gc.collect()
    return output


def build_native_moe(
    backend,
    cpu_infer,
    weight_bank: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
):
    expert_count = len(weight_bank)
    config = kt_kernel_ext.moe.MOEConfig(
        expert_count,
        args.benchmark_top_k,
        args.hidden_size,
        args.intermediate_size,
        0,
    )
    config.max_len = 1
    config.layer_idx = args.layer
    config.pool = cpu_infer.backend_
    config.quant_config.quant_method = "NVFP4"
    config.quant_config.bits = 4
    config.quant_config.group_size = 16
    config.quant_config.zero_point = False

    for projection in ("gate", "up", "down"):
        setattr(
            config,
            f"{projection}_projs",
            [[weights[projection].data_ptr() for weights in weight_bank]],
        )
        setattr(
            config,
            f"{projection}_scales",
            [
                [
                    weights[f"{projection}_scale"].data_ptr()
                    for weights in weight_bank
                ]
            ],
        )
        setattr(
            config,
            f"{projection}_scale2s",
            [
                [
                    weights[f"{projection}_scale_2"].data_ptr()
                    for weights in weight_bank
                ]
            ],
        )

    moe = backend(config)
    physical_to_logical = torch.arange(expert_count, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe, physical_to_logical


def native_forward(
    moe, cpu_infer, inputs: torch.Tensor, top_k: int
) -> torch.Tensor:
    output = torch.empty_like(inputs)
    batch_size = torch.ones(1, dtype=torch.int32)
    expert_ids = torch.arange(top_k, dtype=torch.int64).view(1, top_k)
    routing_weights = torch.zeros((1, top_k), dtype=torch.float32)
    routing_weights[0, 0] = 1.0
    cpu_infer.submit(
        moe.forward_task(
            batch_size.data_ptr(),
            top_k,
            expert_ids.data_ptr(),
            routing_weights.data_ptr(),
            inputs.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return output


def benchmark_native_forward(
    moe,
    cpu_infer,
    inputs: torch.Tensor,
    iterations: int,
    warmup: int,
    hidden_size: int,
    intermediate_size: int,
    expert_count: int,
    top_k: int,
) -> None:
    if iterations <= 0:
        return

    output = torch.empty_like(inputs)
    batch_size = torch.ones(1, dtype=torch.int32)
    expert_ids = [
        torch.arange(start, start + top_k, dtype=torch.int64).view(1, top_k)
        for start in range(0, expert_count, top_k)
    ]
    routing_weights = torch.full(
        (1, top_k), 1.0 / top_k, dtype=torch.float32
    )
    route_count = len(expert_ids)

    def run_once(expert_index: int) -> None:
        cpu_infer.submit(
            moe.forward_task(
                batch_size.data_ptr(),
                top_k,
                expert_ids[expert_index % route_count].data_ptr(),
                routing_weights.data_ptr(),
                inputs.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpu_infer.sync()

    warmup_runs = max(max(warmup, 0), route_count)
    for iteration in range(warmup_runs):
        run_once(iteration)

    started = time.perf_counter()
    for iteration in range(iterations):
        run_once(iteration)
    elapsed = time.perf_counter() - started

    seconds_per_forward = elapsed / iterations
    projection_elements = 3 * hidden_size * intermediate_size
    checkpoint_bytes = projection_elements * (0.5 + 1.0 / 16.0)
    effective_gbps = checkpoint_bytes * top_k / seconds_per_forward / 1e9
    benchmark_kind = "LLC-hot" if expert_count == 1 else "rotating"
    working_set_gb = checkpoint_bytes * expert_count / 1e9
    print(
        f"  benchmark ({benchmark_kind}, experts={expert_count}, top_k={top_k}, "
        f"weight_working_set={working_set_gb:.3f} GB): "
        f"{seconds_per_forward * 1e3:.3f} ms/forward "
        f"{1.0 / seconds_per_forward:.3f} forwards/s "
        f"{effective_gbps:.3f} effective_weight_GB/s"
    )


def selected_backends(name: str):
    requested = {
        "amx": ("AMXFP4_KGroup_MOE",),
        "avx2": ("AVX2MXFP4_MOE",),
        "all": ("AMXFP4_KGroup_MOE", "AVX2MXFP4_MOE"),
    }[name]
    result = []
    for backend_name in requested:
        backend = getattr(kt_kernel_ext.moe, backend_name, None)
        if backend is not None:
            result.append((backend_name, backend))
    if not result:
        raise RuntimeError(f"No requested NVFP4 backend is compiled: {requested}")
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    benchmark_expert_ids, weight_bank = load_checkpoint_experts(args)
    weights = weight_bank[0]
    print(
        f"checkpoint={args.model_path} layer={args.layer} expert={args.expert} "
        f"benchmark_experts={benchmark_expert_ids[0]}.."
        f"{benchmark_expert_ids[-1]}"
    )
    for projection in ("gate", "up", "down"):
        print(
            f"{projection}: weight={tuple(weights[projection].shape)} "
            f"scale={tuple(weights[f'{projection}_scale'].shape)} "
            f"scale_2={weights[f'{projection}_scale_2'].item():.9g}"
        )

    inputs = (torch.randn((1, args.hidden_size)) * 0.05).to(torch.bfloat16)
    expected = reference_forward(inputs, weights)
    expected_mean = expected.float().abs().mean().item()
    print(f"reference mean_abs={expected_mean:.9g}")

    failed_backends = []
    for backend_name, backend in selected_backends(args.backend):
        cpu_infer = make_cpu_infer(args.threads, args.numa_nodes)
        moe, physical_to_logical = build_native_moe(
            backend, cpu_infer, weight_bank, args
        )
        actual = native_forward(
            moe, cpu_infer, inputs, args.benchmark_top_k
        )
        difference = (actual.float() - expected.float()).abs()
        relative_error = difference.mean().item() / (expected_mean + 1e-12)
        passed = bool(torch.isfinite(actual.float()).all()) and (
            relative_error <= args.relative_tolerance
        )
        print(
            f"{backend_name}: relative_error={relative_error:.9g} "
            f"max_abs_error={difference.max().item():.9g} "
            f"finite={torch.isfinite(actual.float()).all().item()} "
            f"status={'PASS' if passed else 'FAIL'}"
        )
        print("  expected[:8]", expected[0, :8].float().tolist())
        print("  actual[:8]  ", actual[0, :8].float().tolist())
        benchmark_native_forward(
            moe,
            cpu_infer,
            inputs,
            args.benchmark_iterations,
            args.benchmark_warmup,
            args.hidden_size,
            args.intermediate_size,
            len(weight_bank),
            args.benchmark_top_k,
        )
        del moe, physical_to_logical, cpu_infer, actual
        gc.collect()
        if not passed:
            failed_backends.append(backend_name)

    if failed_backends:
        raise SystemExit(
            "Checkpoint-native NVFP4 validation failed for: "
            + ", ".join(failed_backends)
        )


if __name__ == "__main__":
    main()
