"""Compare a real checkpoint NVFP4 expert with a PyTorch reference."""

from __future__ import annotations

import argparse
import gc
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


def load_checkpoint_expert(args: argparse.Namespace):
    loader = NVFP4SafeTensorLoader(str(args.model_path))
    weights = loader.load_experts(
        f"model.layers.{args.layer}", expert_ids=[args.expert]
    )
    return {
        name: values[args.expert]
        for name, values in weights.items()
    }


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
    weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
):
    config = kt_kernel_ext.moe.MOEConfig(
        1,
        1,
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
        setattr(config, f"{projection}_projs", [[weights[projection].data_ptr()]])
        setattr(
            config,
            f"{projection}_scales",
            [[weights[f"{projection}_scale"].data_ptr()]],
        )
        setattr(
            config,
            f"{projection}_scale2s",
            [[weights[f"{projection}_scale_2"].data_ptr()]],
        )

    moe = backend(config)
    physical_to_logical = torch.zeros(1, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe, physical_to_logical


def native_forward(moe, cpu_infer, inputs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(inputs)
    batch_size = torch.ones(1, dtype=torch.int32)
    expert_ids = torch.zeros((1, 1), dtype=torch.int64)
    routing_weights = torch.ones((1, 1), dtype=torch.float32)
    cpu_infer.submit(
        moe.forward_task(
            batch_size.data_ptr(),
            1,
            expert_ids.data_ptr(),
            routing_weights.data_ptr(),
            inputs.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return output


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

    weights = load_checkpoint_expert(args)
    print(
        f"checkpoint={args.model_path} layer={args.layer} expert={args.expert}"
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
            backend, cpu_infer, weights, args
        )
        actual = native_forward(moe, cpu_infer, inputs)
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
