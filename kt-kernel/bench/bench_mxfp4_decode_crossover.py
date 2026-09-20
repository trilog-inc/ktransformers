#!/usr/bin/env python3
"""Benchmark the MXFP4 AVX512/AMX crossover at decode-sized MoE shapes."""

import argparse
import json
import statistics
import time

import torch

from kt_kernel import kt_kernel_ext


def make_source(hidden_size: int, intermediate_size: int, random_source: bool):
    # E2M1 nibble 2 is +1.0. Reusing one source pointer for every logical
    # expert keeps benchmark setup small; the native loader still builds the
    # same NUMA-local compact banks used by inference.
    if random_source:
        generator = torch.Generator().manual_seed(20260920)
        gate = torch.randint(
            0,
            256,
            (intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
        up = torch.randint(
            0,
            256,
            (intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
        down = torch.randint(
            0,
            256,
            (hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            generator=generator,
        )
        gate_scale = (
            torch.rand(
                (intermediate_size, hidden_size // 32), generator=generator
            )
            * 0.09
            + 0.01
        ).bfloat16()
    else:
        packed = 0x22
        gate = torch.full(
            (intermediate_size, hidden_size // 2), packed, dtype=torch.uint8
        )
        up = gate.clone()
        down = torch.full(
            (hidden_size, intermediate_size // 2), packed, dtype=torch.uint8
        )
        gate_scale = torch.ones(
            (intermediate_size, hidden_size // 32), dtype=torch.bfloat16
        )
    up_scale = gate_scale.clone()
    if random_source:
        down_scale = (
            torch.rand(
                (hidden_size, intermediate_size // 32), generator=generator
            )
            * 0.09
            + 0.01
        ).bfloat16()
    else:
        down_scale = torch.ones(
            (hidden_size, intermediate_size // 32), dtype=torch.bfloat16
        )
    return gate, up, down, gate_scale, up_scale, down_scale


def build_moe(args):
    source = make_source(
        args.hidden_size, args.intermediate_size, args.random_source
    )
    sources = [source] * args.experts
    config = kt_kernel_ext.moe.MOEConfig(
        args.experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        0,
    )
    cpu_infer = kt_kernel_ext.CPUInfer(args.threads)
    config.pool = cpu_infer.backend_
    config.max_len = args.tokens
    config.quant_config.bits = 4
    config.quant_config.group_size = 32
    config.quant_config.zero_point = False
    config.gate_projs = [[item[0].data_ptr() for item in sources]]
    config.up_projs = [[item[1].data_ptr() for item in sources]]
    config.down_projs = [[item[2].data_ptr() for item in sources]]
    config.gate_scales = [[item[3].data_ptr() for item in sources]]
    config.up_scales = [[item[4].data_ptr() for item in sources]]
    config.down_scales = [[item[5].data_ptr() for item in sources]]
    physical_to_logical = torch.arange(args.experts, dtype=torch.int64)
    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return cpu_infer, moe, source, physical_to_logical


def make_routes(args):
    if args.routes == "distinct":
        ids = torch.arange(args.tokens * args.topk, dtype=torch.int64)
        ids = ids.remainder(args.experts).reshape(args.tokens, args.topk)
    elif args.routes == "repeated":
        ids = torch.arange(args.topk, dtype=torch.int64).repeat(args.tokens, 1)
    else:
        raise ValueError(args.routes)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=289)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--threads", type=int, default=28)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--routes", choices=("distinct", "repeated"), default="distinct")
    parser.add_argument("--rotate-routes", action="store_true")
    parser.add_argument("--random-source", action="store_true")
    args = parser.parse_args()

    cpu_infer, moe, source, physical_to_logical = build_moe(args)
    qlen = torch.tensor([args.tokens], dtype=torch.int64)
    expert_ids = make_routes(args)
    expert_weights = torch.full(
        (args.tokens, args.topk), 1.0 / args.topk, dtype=torch.float32
    )
    hidden_states = torch.full(
        (args.tokens, args.hidden_size), 0.001, dtype=torch.bfloat16
    )
    output = torch.empty_like(hidden_states)

    base_expert_ids = expert_ids.clone()

    def set_routes(step: int):
        if args.rotate_routes:
            expert_ids.copy_((base_expert_ids + step * args.topk).remainder(args.experts))

    def forward():
        cpu_infer.submit(
            moe.forward_task(
                qlen.data_ptr(),
                1,
                expert_ids.data_ptr(),
                expert_weights.data_ptr(),
                hidden_states.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        cpu_infer.sync()

    for step in range(args.warmup):
        set_routes(step)
        forward()
    samples = []
    for step in range(args.rounds):
        set_routes(args.warmup + step)
        begin = time.perf_counter()
        forward()
        samples.append((time.perf_counter() - begin) * 1000.0)

    ordered = sorted(samples)
    result = {
        "routes": args.routes,
        "tokens": args.tokens,
        "topk": args.topk,
        "rotate_routes": args.rotate_routes,
        "random_source": args.random_source,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "checksum": float(output.float().sum()),
        "finite": bool(torch.isfinite(output).all()),
    }
    print(json.dumps(result, sort_keys=True))

    # Keep pointer owners alive through the final native synchronization.
    assert source and physical_to_logical.numel() == args.experts


if __name__ == "__main__":
    main()
