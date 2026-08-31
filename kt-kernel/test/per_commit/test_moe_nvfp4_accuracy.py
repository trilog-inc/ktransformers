"""Numerical contracts for checkpoint-native ModelOpt NVFP4 CPU experts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default")

import pytest
import torch
import kt_kernel_ext


EXPERT_NUM = 4
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
TOP_K = 2
MAX_LEN = 16
GROUP_SIZE = 16

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


def _available_backends():
    flags = set()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith(("flags", "Features")):
                    flags.update(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass

    names = ["AVX2MXFP4_MOE"]
    if {"avx512f", "avx512bw", "avx512_bf16"}.issubset(flags):
        names.append("AMXFP4_KGroup_MOE")
    return [
        (name, getattr(kt_kernel_ext.moe, name))
        for name in names
        if hasattr(kt_kernel_ext.moe, name)
    ]


def _make_cpu_infer():
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [0]
    config.subpool_thread_count = [4]
    return kt_kernel_ext.CPUInfer(config)


def _make_projection(experts, rows, columns, seed, tensor_scales):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(
        0, 16, (experts, rows, columns), dtype=torch.uint8, generator=generator
    )
    packed = ((codes[..., 1::2] << 4) | codes[..., 0::2]).contiguous()

    scale_choices = torch.tensor([0.03125, 0.0625, 0.125, 0.25], dtype=torch.float32)
    scale_indices = torch.randint(
        0,
        len(scale_choices),
        (experts, rows, columns // GROUP_SIZE),
        generator=generator,
    )
    block_scales = scale_choices[scale_indices].to(torch.float8_e4m3fn).contiguous()
    scale_2 = torch.tensor(tensor_scales, dtype=torch.float32).reshape(experts, 1)

    values = E2M1[codes.long()].view(experts, rows, columns // GROUP_SIZE, GROUP_SIZE)
    dequantized = (
        values * block_scales.float().unsqueeze(-1) * scale_2[:, None, None, :]
    ).reshape(experts, rows, columns)
    return packed, block_scales, scale_2.contiguous(), dequantized


def _make_weights():
    gate = _make_projection(
        EXPERT_NUM,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        11,
        [0.5, 0.75, 1.0, 1.25],
    )
    up = _make_projection(
        EXPERT_NUM,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        17,
        [1.5, 1.25, 1.0, 0.75],
    )
    down = _make_projection(
        EXPERT_NUM,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        23,
        [0.625, 0.875, 1.125, 1.375],
    )
    return gate, up, down


def _build_moe(backend, cpu_infer, projections):
    gate, up, down = projections
    config = kt_kernel_ext.moe.MOEConfig(
        EXPERT_NUM, TOP_K, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0
    )
    config.max_len = MAX_LEN
    config.pool = cpu_infer.backend_
    config.quant_config.quant_method = "NVFP4"
    config.quant_config.bits = 4
    config.quant_config.group_size = GROUP_SIZE
    config.quant_config.zero_point = False

    for name, projection in zip(("gate", "up", "down"), projections):
        packed, block_scale, scale_2, _ = projection
        setattr(config, f"{name}_projs", [[t.data_ptr() for t in packed]])
        setattr(config, f"{name}_scales", [[t.data_ptr() for t in block_scale]])
        setattr(config, f"{name}_scale2s", [[t.data_ptr() for t in scale_2]])

    moe = backend(config)
    physical_to_logical = torch.arange(EXPERT_NUM, dtype=torch.int64)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    return moe


def _reference_moe(inputs, expert_ids, routing_weights, projections):
    gate, up, down = (projection[3] for projection in projections)
    output = torch.zeros(inputs.shape[0], HIDDEN_SIZE, dtype=torch.float32)
    for token in range(inputs.shape[0]):
        token_input = inputs[token : token + 1].float()
        for slot in range(TOP_K):
            expert = int(expert_ids[token, slot])
            gate_out = torch.mm(token_input, gate[expert].t()).to(torch.bfloat16)
            up_out = torch.mm(token_input, up[expert].t()).to(torch.bfloat16)
            activated = (
                torch.nn.functional.silu(gate_out.float()) * up_out.float()
            ).to(torch.bfloat16)
            expert_out = torch.mm(activated.float(), down[expert].t())
            output[token] += expert_out[0] * routing_weights[token, slot]
    return output.to(torch.bfloat16)


@pytest.mark.cpu
@pytest.mark.parametrize("qlen", [1, 8])
@pytest.mark.parametrize("backend_name,backend", _available_backends())
def test_native_nvfp4_matches_dequantized_reference(backend_name, backend, qlen):
    projections = _make_weights()
    cpu_infer = _make_cpu_infer()
    moe = _build_moe(backend, cpu_infer, projections)

    generator = torch.Generator().manual_seed(100 + qlen)
    inputs = (torch.randn((qlen, HIDDEN_SIZE), generator=generator) * 0.05).to(
        torch.bfloat16
    )
    expert_ids = torch.stack(
        [torch.randperm(EXPERT_NUM, generator=generator)[:TOP_K] for _ in range(qlen)]
    ).to(torch.int64)
    routing_weights = torch.rand(
        (qlen, TOP_K), dtype=torch.float32, generator=generator
    )
    output = torch.empty((qlen, HIDDEN_SIZE), dtype=torch.bfloat16)
    batch_size = torch.tensor([qlen], dtype=torch.int32)

    cpu_infer.submit(
        moe.forward_task(
            batch_size.data_ptr(),
            TOP_K,
            expert_ids.data_ptr(),
            routing_weights.data_ptr(),
            inputs.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()

    expected = _reference_moe(inputs, expert_ids, routing_weights, projections)
    relative_error = torch.mean(torch.abs(output.float() - expected.float())) / (
        torch.mean(torch.abs(expected.float())) + 1e-8
    )
    assert torch.isfinite(output.float()).all()
    assert (
        relative_error < 0.08
    ), f"{backend_name} NVFP4 relative error {relative_error.item():.6f} exceeds 0.08"
